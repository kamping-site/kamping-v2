// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

#include "kamping/kassert/kassert.hpp"
#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/view_interface.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {

/// Dependent false for static_assert in if-constexpr chains (in C++20 the condition
/// must depend on a template parameter, otherwise the assertion fires unconditionally).
template <typename>
inline constexpr bool always_false = false;

template <typename R>
concept nested_send_buffer = std::ranges::forward_range<R> && std::ranges::sized_range<R>
                             && std::ranges::input_range<std::ranges::range_value_t<R>>
                             && std::ranges::sized_range<std::ranges::range_value_t<R>>;

template <typename T>
concept pair_like = requires(T t) { std::tuple_size<T>::value == 2; };

template <typename T>
concept destination_buffer_pair =
    pair_like<T> && mpi::experimental::rank<std::tuple_element_t<0, T>>
    && std::ranges::input_range<std::tuple_element_t<1, T>> && std::ranges::sized_range<std::tuple_element_t<1, T>>;

template <typename R>
concept sparse_nested_send_buffer =
    std::ranges::forward_range<R> && destination_buffer_pair<std::ranges::range_value_t<R>>;

// The value slot (element 0) must not itself be a range. A range-valued payload means
// "multiple elements for this rank" and belongs to the buffer path
// (destination_buffer_pair), not the single-value path. Without this exclusion a
// (range, rank) pair would match here and the whole range would be copied into a single
// element slot. Such a pair now matches no flattenable concept and fails to compile.
template <typename T>
concept value_destination_pair = pair_like<T> && mpi::experimental::rank<std::tuple_element_t<1, T>>
                                 && !std::ranges::input_range<std::tuple_element_t<0, T>>;

template <typename R>
concept value_destination_pair_buffer =
    std::ranges::forward_range<R> && value_destination_pair<std::ranges::range_value_t<R>>;

template <typename R>
struct flat_element {};

template <nested_send_buffer R>
struct flat_element<R> {
    using type = std::ranges::range_value_t<std::ranges::range_value_t<R>>;
};

template <sparse_nested_send_buffer R>
struct flat_element<R> {
    using type = std::ranges::range_value_t<std::tuple_element_t<1, std::ranges::range_value_t<R>>>;
};

template <value_destination_pair_buffer R>
struct flat_element<R> {
    using type = std::tuple_element_t<0, std::ranges::range_value_t<R>>;
};

template <typename R>
using flat_element_t = flat_element<R>::type;

template <typename R>
concept flattenable_send_buffer =
    nested_send_buffer<R> || sparse_nested_send_buffer<R> || value_destination_pair_buffer<R>;

/// Flattens a range-of-ranges into a contiguous MPI buffer with per-rank counts
/// and displacements.
///
/// The source range-of-ranges is lazily flattened (copied) into the flat buffer
/// on first access to any MPI protocol method. Counts are derived from inner
/// range sizes, displacements via exclusive scan.
///
/// mpi_type/mpi_count/mpi_ptr are forwarded from base() (= flat_buf_) via
/// view_interface. Calling base() triggers ensure_flattened(), so the flat buffer
/// is always fully populated before any accessor reads it.
///
/// mpi_counts() and mpi_displs() are provided explicitly and are also lazy
/// (same ensure_flattened() guard).
///
/// Template parameters:
///   Source        — the range-of-ranges (after all() wrapping)
///   FlatBuf       — the flat data buffer (after all() wrapping)
///   Counts        — the counts container (after all() wrapping)
///   Displs        — the displacements container (after all() wrapping)
///   resize_buf    — if true, flat buffer is resized as needed
///   resize_counts — if true, counts buffer is resized as needed
///   resize_displs — if true, displs buffer is resized as needed
///
/// Typical usage:
///   std::vector<std::vector<int>> per_rank = ...;
///   kamping::v2::alltoallv(per_rank | flatten_v(), rbuf);
template <
    typename Source,
    typename FlatBuf,
    mpi::experimental::count_range Counts,
    mpi::experimental::count_range Displs,
    bool                           resize_buf    = false,
    bool                           resize_counts = false,
    bool                           resize_displs = false>
    requires flattenable_send_buffer<Source>
             && (!resize_buf || has_resize<FlatBuf> || has_mpi_resize_for_receive<FlatBuf>)
             && (!resize_counts || has_resize<Counts> || has_mpi_resize_for_receive<Counts>)
             && (!resize_displs || has_resize<Displs> || has_mpi_resize_for_receive<Displs>)
class flatten_v_view
    : public view_interface<flatten_v_view<Source, FlatBuf, Counts, Displs, resize_buf, resize_counts, resize_displs>> {
    Source          source_;
    mutable FlatBuf flat_buf_;
    mutable Counts  counts_;
    mutable Displs  displs_;
    mutable bool    needs_flatten_ = true;
    // Communicator size, supplied via set_comm_size() through the infer() protocol.
    // Required to lay out counts/displs for sparse sources (where the source does not
    // span all ranks); unused for dense nested_send_buffer sources.
    mutable std::optional<int> comm_size_;

    void ensure_flattened() const {
        if (!needs_flatten_)
            return;

        // The counts/displs arrays span one entry per rank. For a dense nested buffer
        // there is exactly one inner range per rank, so the source size is the rank
        // count. Sparse sources only mention a subset of ranks, so the communicator
        // size must have been provided via set_comm_size().
        std::size_t num_ranks;
        if constexpr (nested_send_buffer<Source>) {
            num_ranks = std::ranges::size(source_);
        } else {
            KAMPING_ASSERT(
                comm_size_.has_value(),
                "set_comm_size() must be called before flattening a sparse send buffer"
            );
            num_ranks = static_cast<std::size_t>(*comm_size_);
        }

        // Resize and fill counts
        if constexpr (resize_counts) {
            kamping::v2::resize_for_receive(counts_, static_cast<std::ptrdiff_t>(num_ranks));
        }
        auto*          counts_ptr = std::ranges::data(counts_);
        std::ptrdiff_t total_size = 0;
        if constexpr (nested_send_buffer<Source>) {
            std::size_t idx = 0;
            for (auto&& inner: source_) {
                auto const s      = static_cast<int>(std::ranges::size(inner));
                counts_ptr[idx++] = s;
                total_size += s;
            }
        } else {
            // Ranks may be mentioned out of order, repeatedly, or not at all.
            std::ranges::fill(counts_ptr, counts_ptr + num_ranks, 0);
            for (auto&& [first, second]: source_) {
                if constexpr (sparse_nested_send_buffer<Source>) {
                    auto const r = mpi::experimental::to_rank(first);
                    auto const s = static_cast<int>(std::ranges::size(second));
                    counts_ptr[r] += s;
                    total_size += s;
                } else if constexpr (value_destination_pair_buffer<Source>) {
                    auto const r = mpi::experimental::to_rank(second);
                    counts_ptr[r] += 1;
                    total_size += 1;
                } else {
                    static_assert(always_false<Source>, "unhandled flattenable_send_buffer kind");
                }
            }
        }

        // Resize and compute displacements
        if constexpr (resize_displs) {
            kamping::v2::resize_for_receive(displs_, static_cast<std::ptrdiff_t>(num_ranks));
        }
        auto* displs_ptr = std::ranges::data(displs_);
        std::exclusive_scan(counts_ptr, counts_ptr + num_ranks, displs_ptr, 0);

        // Resize and copy data into flat buffer
        if constexpr (resize_buf) {
            kamping::v2::resize_for_receive(flat_buf_, total_size);
        }
        using elem_t      = flat_element_t<Source>;
        elem_t* flat_data = std::ranges::data(flat_buf_);
        if constexpr (nested_send_buffer<Source>) {
            elem_t* dest = flat_data;
            for (auto&& inner: source_) {
                dest = std::ranges::copy(inner, dest).out;
            }
        } else {
            // Scatter each pair into its destination rank's slice. The displacements
            // double as per-rank write cursors (handling out-of-order and repeated
            // ranks), avoiding a separate cursor allocation.
            for (auto&& [first, second]: source_) {
                if constexpr (sparse_nested_send_buffer<Source>) {
                    auto const r = mpi::experimental::to_rank(first);
                    std::ranges::copy(second, flat_data + displs_ptr[r]);
                    displs_ptr[r] += static_cast<int>(std::ranges::size(second));
                } else if constexpr (value_destination_pair_buffer<Source>) {
                    auto const r               = mpi::experimental::to_rank(second);
                    flat_data[displs_ptr[r]++] = first;
                } else {
                    static_assert(always_false<Source>, "unhandled flattenable_send_buffer kind");
                }
            }
            // Each cursor now holds its rank's end offset (= the next rank's start), so
            // the exclusive-scan displacements are just the cursors shifted right by one,
            // with displs[0] = 0 (there is always at least one rank).
            std::shift_right(std::ranges::begin(displs_), std::ranges::end(displs_), 1);
            displs_ptr[0] = 0;
        }

        needs_flatten_ = false;
    }

public:
    /// Returns the flat data buffer, triggering lazy flattening on first call.
    /// view_interface forwards mpi_count/mpi_ptr/mpi_type from here automatically.
    FlatBuf const& base() const& {
        ensure_flattened();
        return flat_buf_;
    }
    FlatBuf& base() & {
        ensure_flattened();
        return flat_buf_;
    }

    template <typename S, typename F, typename C, typename D>
    flatten_v_view(S&& source, F&& flat_buf, C&& counts, D&& displs)
        : source_(kamping::v2::all(std::forward<S>(source))),
          flat_buf_(kamping::v2::all(std::forward<F>(flat_buf))),
          counts_(kamping::v2::all(std::forward<C>(counts))),
          displs_(kamping::v2::all(std::forward<D>(displs))) {}

    /// Supplies the communicator size so that sparse sources can lay out their
    /// counts/displs/data over all ranks. Called by the infer() protocol (see
    /// deferred_send_buf_v) before any accessor is read. A no-op effect for dense
    /// nested sources, whose rank count is derived from the source size.
    void set_comm_size(int n) const {
        comm_size_     = n;
        needs_flatten_ = true;
    }

    std::span<int const> mpi_counts() const {
        ensure_flattened();
        return {std::ranges::data(counts_), std::ranges::size(counts_)};
    }

    std::span<int const> mpi_displs() const {
        ensure_flattened();
        return {std::ranges::data(displs_), std::ranges::size(displs_)};
    }

    /// Displacements are always computed via exclusive_scan — monotonically non-decreasing.
    constexpr bool displs_monotonic() const {
        return true;
    }

    constexpr Counts const& counts() const& {
        return counts_;
    }
    constexpr Counts& counts() & {
        return counts_;
    }
    constexpr Counts&& counts() && {
        return std::move(counts_);
    }

    constexpr Displs const& displs() const& {
        return displs_;
    }
    constexpr Displs& displs() & {
        return displs_;
    }
    constexpr Displs&& displs() && {
        return std::move(displs_);
    }
};

template <typename S, typename F, typename C, typename D>
flatten_v_view(S&&, F&&, C&&, D&&)
    -> flatten_v_view<kamping::v2::all_t<S>, kamping::v2::all_t<F>, kamping::v2::all_t<C>, kamping::v2::all_t<D>>;

template <typename Source, typename FlatBuf, typename Counts, typename Displs, bool rb, bool rc, bool rd>
inline constexpr bool enable_borrowed_buffer<flatten_v_view<Source, FlatBuf, Counts, Displs, rb, rc, rd>> =
    enable_borrowed_buffer<FlatBuf> && enable_borrowed_buffer<Counts> && enable_borrowed_buffer<Displs>;

} // namespace kamping::v2

namespace kamping::v2::views {

/// 0-arg: allocate flat buffer, counts, and displs internally — all auto-resized.
/// FlatTemplate/CountsContainer/DisplsContainer select the internal container types:
///   flatten_v<std::deque>()                       — custom flat buffer container
///   flatten_v<std::vector, std::deque<int>>()     — custom counts container
template <
    template <typename...> typename FlatTemplate = std::vector,
    typename CountsContainer                     = std::vector<int>,
    typename DisplsContainer                     = std::vector<int>>
constexpr auto flatten_v() {
    return kamping::v2::adaptor<0, decltype([](auto&& source) {
                                    using elem_t = flat_element_t<std::remove_cvref_t<decltype(source)>>;
                                    using S      = kamping::v2::all_t<decltype(source)>;
                                    using F      = kamping::v2::owning_view<FlatTemplate<elem_t>>;
                                    using C      = kamping::v2::owning_view<CountsContainer>;
                                    using D      = kamping::v2::owning_view<DisplsContainer>;
                                    return kamping::v2::flatten_v_view<S, F, C, D, true, true, true>(
                                        std::forward<decltype(source)>(source),
                                        FlatTemplate<elem_t>{},
                                        CountsContainer{},
                                        DisplsContainer{}
                                    );
                                })>{}();
}

/// 1-arg (non-resize): user flat buffer, internal counts and displs (auto-resized).
template <typename CountsContainer = std::vector<int>, typename DisplsContainer = std::vector<int>, typename Fb>
    requires(!std::same_as<std::remove_cvref_t<Fb>, kamping::v2::resize_t>)
constexpr auto flatten_v(Fb&& flat_buf) {
    return kamping::v2::adaptor<1, decltype([](auto&& source, auto&& fb) {
                                    using S = kamping::v2::all_t<decltype(source)>;
                                    using F = kamping::v2::all_t<decltype(fb)>;
                                    using C = kamping::v2::owning_view<CountsContainer>;
                                    using D = kamping::v2::owning_view<DisplsContainer>;
                                    return kamping::v2::flatten_v_view<S, F, C, D, false, true, true>(
                                        std::forward<decltype(source)>(source),
                                        std::forward<decltype(fb)>(fb),
                                        CountsContainer{},
                                        DisplsContainer{}
                                    );
                                })>{}(std::forward<Fb>(flat_buf));
}

/// 2-arg (non-resize): user flat buffer + user counts, internal displs (auto-resized).
template <typename DisplsContainer = std::vector<int>, typename Fb, typename Ct>
    requires(!std::same_as<std::remove_cvref_t<Fb>, kamping::v2::resize_t>)
constexpr auto flatten_v(Fb&& flat_buf, Ct&& counts) {
    return kamping::v2::adaptor<2, decltype([](auto&& source, auto&& fb, auto&& c) {
                                    using S = kamping::v2::all_t<decltype(source)>;
                                    using F = kamping::v2::all_t<decltype(fb)>;
                                    using C = kamping::v2::all_t<decltype(c)>;
                                    using D = kamping::v2::owning_view<DisplsContainer>;
                                    return kamping::v2::flatten_v_view<S, F, C, D, false, false, true>(
                                        std::forward<decltype(source)>(source),
                                        std::forward<decltype(fb)>(fb),
                                        std::forward<decltype(c)>(c),
                                        DisplsContainer{}
                                    );
                                })>{}(std::forward<Fb>(flat_buf), std::forward<Ct>(counts));
}

/// 3-arg (non-resize): user flat buffer + user counts + user displs — no resize for any.
template <typename Fb, typename Ct, typename Dt>
    requires(!std::same_as<std::remove_cvref_t<Fb>, kamping::v2::resize_t>)
constexpr auto flatten_v(Fb&& flat_buf, Ct&& counts, Dt&& displs) {
    return kamping::v2::adaptor<3, decltype([](auto&& source, auto&& fb, auto&& c, auto&& d) {
                                    return kamping::v2::flatten_v_view(
                                        std::forward<decltype(source)>(source),
                                        std::forward<decltype(fb)>(fb),
                                        std::forward<decltype(c)>(c),
                                        std::forward<decltype(d)>(d)
                                    );
                                })>{}(std::forward<Fb>(flat_buf), std::forward<Ct>(counts), std::forward<Dt>(displs));
}

/// 1-arg (resize): user flat buffer with resize, internal counts and displs (auto-resized).
template <typename CountsContainer = std::vector<int>, typename DisplsContainer = std::vector<int>, typename Fb>
constexpr auto flatten_v(kamping::v2::resize_t, Fb&& flat_buf) {
    return kamping::v2::adaptor<1, decltype([](auto&& source, auto&& fb) {
                                    using S = kamping::v2::all_t<decltype(source)>;
                                    using F = kamping::v2::all_t<decltype(fb)>;
                                    using C = kamping::v2::owning_view<CountsContainer>;
                                    using D = kamping::v2::owning_view<DisplsContainer>;
                                    return kamping::v2::flatten_v_view<S, F, C, D, true, true, true>(
                                        std::forward<decltype(source)>(source),
                                        std::forward<decltype(fb)>(fb),
                                        CountsContainer{},
                                        DisplsContainer{}
                                    );
                                })>{}(std::forward<Fb>(flat_buf));
}

/// 2-arg (resize): user flat buffer + user counts — both resized, internal displs (auto-resized).
template <typename DisplsContainer = std::vector<int>, typename Fb, typename Ct>
constexpr auto flatten_v(kamping::v2::resize_t, Fb&& flat_buf, Ct&& counts) {
    return kamping::v2::adaptor<2, decltype([](auto&& source, auto&& fb, auto&& c) {
                                    using S = kamping::v2::all_t<decltype(source)>;
                                    using F = kamping::v2::all_t<decltype(fb)>;
                                    using C = kamping::v2::all_t<decltype(c)>;
                                    using D = kamping::v2::owning_view<DisplsContainer>;
                                    return kamping::v2::flatten_v_view<S, F, C, D, true, true, true>(
                                        std::forward<decltype(source)>(source),
                                        std::forward<decltype(fb)>(fb),
                                        std::forward<decltype(c)>(c),
                                        DisplsContainer{}
                                    );
                                })>{}(std::forward<Fb>(flat_buf), std::forward<Ct>(counts));
}

/// 3-arg (resize): user flat buffer + user counts + user displs — all resized.
template <typename Fb, typename Ct, typename Dt>
constexpr auto flatten_v(kamping::v2::resize_t, Fb&& flat_buf, Ct&& counts, Dt&& displs) {
    return kamping::v2::adaptor<3, decltype([](auto&& source, auto&& fb, auto&& c, auto&& d) {
                                    using S = kamping::v2::all_t<decltype(source)>;
                                    using F = kamping::v2::all_t<decltype(fb)>;
                                    using C = kamping::v2::all_t<decltype(c)>;
                                    using D = kamping::v2::all_t<decltype(d)>;
                                    return kamping::v2::flatten_v_view<S, F, C, D, true, true, true>(
                                        std::forward<decltype(source)>(source),
                                        std::forward<decltype(fb)>(fb),
                                        std::forward<decltype(c)>(c),
                                        std::forward<decltype(d)>(d)
                                    );
                                })>{}(std::forward<Fb>(flat_buf), std::forward<Ct>(counts), std::forward<Dt>(displs));
}

} // namespace kamping::v2::views
