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

#include "kamping/v2/kassert.hpp"
#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/payload.hpp"
#include "kamping/v2/views/view_interface.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {

namespace detail {
/// Dependent false for static_assert in if-constexpr chains.
template <typename>
inline constexpr bool always_false = false;
} // namespace detail

/// Flattens a structured send buffer (see copyable_send_buffer) into a contiguous MPI buffer
/// with per-rank counts and displacements.
///
/// The source is lazily flattened (copied) into the flat buffer on first access to any MPI
/// protocol method. Counts are derived from the message sizes, displacements via exclusive scan.
/// Sparse sources need the communicator size, which infer() supplies via set_comm_size().
///
/// mpi_type/mpi_count/mpi_ptr are forwarded from base() (= _flat_buf) via
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
    requires copyable_send_buffer<Source> && (!resize_buf || has_resize<FlatBuf> || has_mpi_resize_for_receive<FlatBuf>)
             && (!resize_counts || has_resize<Counts> || has_mpi_resize_for_receive<Counts>)
             && (!resize_displs || has_resize<Displs> || has_mpi_resize_for_receive<Displs>)
class flatten_v_view
    : public view_interface<flatten_v_view<Source, FlatBuf, Counts, Displs, resize_buf, resize_counts, resize_displs>> {
    Source          _source;
    mutable FlatBuf _flat_buf;
    mutable Counts  _counts;
    mutable Displs  _displs;
    mutable bool    _needs_flatten = true;
    // Communicator size, supplied via set_comm_size() through the infer() protocol.
    // Required to lay out counts/displs for sparse sources (where the source does not
    // span all ranks); unused for dense nested_send_buffer sources.
    mutable std::optional<int> _comm_size;

    void ensure_flattened() const {
        if (!_needs_flatten)
            return;

        // The counts/displs arrays span one entry per rank. For a dense nested buffer
        // there is exactly one inner range per rank, so the source size is the rank
        // count. Sparse sources only mention a subset of ranks, so the communicator
        // size must have been provided via set_comm_size().
        std::size_t num_ranks;
        if constexpr (nested_send_buffer<Source>) {
            num_ranks = static_cast<std::size_t>(std::ranges::size(_source));
        } else {
            KAMPING_V2_ASSERT(
                _comm_size.has_value(),
                "set_comm_size() must be called before flattening a sparse send buffer"
            );
            num_ranks = static_cast<std::size_t>(*_comm_size);
        }

        // Resize and fill counts
        if constexpr (resize_counts) {
            kamping::v2::resize_for_receive(_counts, static_cast<std::ptrdiff_t>(num_ranks));
        }
        auto*          counts_ptr = std::ranges::data(_counts);
        std::ptrdiff_t total_size = 0;
        if constexpr (nested_send_buffer<Source>) {
            std::size_t idx = 0;
            for (auto&& inner: _source) {
                auto const s      = static_cast<int>(std::ranges::size(inner));
                counts_ptr[idx++] = s;
                total_size += s;
            }
        } else {
            // Ranks may be mentioned out of order, repeatedly, or not at all.
            std::ranges::fill(counts_ptr, counts_ptr + num_ranks, 0);
            for (auto&& [first, second]: _source) {
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
                    static_assert(detail::always_false<Source>, "unhandled flattenable_send_buffer kind");
                }
            }
        }

        // Resize and compute displacements
        if constexpr (resize_displs) {
            kamping::v2::resize_for_receive(_displs, static_cast<std::ptrdiff_t>(num_ranks));
        }
        auto* displs_ptr = std::ranges::data(_displs);
        std::exclusive_scan(counts_ptr, counts_ptr + num_ranks, displs_ptr, 0);

        // Resize and copy data into flat buffer
        if constexpr (resize_buf) {
            kamping::v2::resize_for_receive(_flat_buf, total_size);
        }
        using elem_t      = payload_element_t<Source>;
        elem_t* flat_data = std::ranges::data(_flat_buf);
        if constexpr (nested_send_buffer<Source>) {
            elem_t* dest = flat_data;
            for (auto&& inner: _source) {
                dest = std::ranges::copy(inner, dest).out;
            }
        } else {
            // Scatter each pair into its destination rank's slice. The displacements
            // double as per-rank write cursors (handling out-of-order and repeated
            // ranks), avoiding a separate cursor allocation.
            for (auto&& [first, second]: _source) {
                if constexpr (sparse_nested_send_buffer<Source>) {
                    auto const r = mpi::experimental::to_rank(first);
                    std::ranges::copy(second, flat_data + displs_ptr[r]);
                    displs_ptr[r] += static_cast<int>(std::ranges::size(second));
                } else if constexpr (value_destination_pair_buffer<Source>) {
                    auto const r               = mpi::experimental::to_rank(second);
                    flat_data[displs_ptr[r]++] = first;
                } else {
                    static_assert(detail::always_false<Source>, "unhandled flattenable_send_buffer kind");
                }
            }
            // Each cursor now holds its rank's end offset (= the next rank's start), so
            // the exclusive-scan displacements are just the cursors shifted right by one,
            // with displs[0] = 0 (there is always at least one rank).
            std::shift_right(std::ranges::begin(_displs), std::ranges::end(_displs), 1);
            displs_ptr[0] = 0;
        }

        _needs_flatten = false;
    }

public:
    /// Returns the flat data buffer, triggering lazy flattening on first call.
    /// view_interface forwards mpi_count/mpi_ptr/mpi_type from here automatically.
    FlatBuf const& base() const& {
        ensure_flattened();
        return _flat_buf;
    }
    FlatBuf& base() & {
        ensure_flattened();
        return _flat_buf;
    }

    /// Reports the flattened buffer's element type. If Source carries a value_type()
    /// annotation (via views::with_value_type / with_auto_value_pool), that payload type is
    /// honored here — the sender's per-element payload, not the deduced flat element type.
    /// Otherwise falls back to FlatBuf's own type (builtin deduction or with_type on FlatBuf).
    constexpr auto mpi_type() const
        requires kamping::v2::has_mpi_value_type<Source> || mpi::experimental::has_mpi_type<FlatBuf>
    {
        if constexpr (kamping::v2::has_mpi_value_type<Source>) {
            return kamping::v2::value_type(_source);
        } else {
            return mpi::experimental::type(base());
        }
    }

    /// Hides view_interface's mpi_value_type() forwarder: a flatten_v_view is a flat buffer that
    /// consumes the value-type annotation of its source (folded into mpi_type() above), so it
    /// has no value-type channel of its own.
    void mpi_value_type() const = delete;

    template <typename S, typename F, typename C, typename D>
    flatten_v_view(S&& source, F&& flat_buf, C&& counts, D&& displs)
        : _source(kamping::v2::all(std::forward<S>(source))),
          _flat_buf(kamping::v2::all(std::forward<F>(flat_buf))),
          _counts(kamping::v2::all(std::forward<C>(counts))),
          _displs(kamping::v2::all(std::forward<D>(displs))) {}

    /// Supplies the communicator size so that sparse sources can lay out their
    /// counts/displs/data over all ranks. Called by the infer() protocol (see
    /// deferred_send_buf_v) before any accessor is read. A no-op effect for dense
    /// nested sources, whose rank count is derived from the source size.
    void set_comm_size(int n) const {
        _comm_size     = n;
        _needs_flatten = true;
    }

    /// @return The per-rank counts, flattening the source first if needed.
    std::span<int const> mpi_counts() const {
        ensure_flattened();
        return {std::ranges::data(_counts), static_cast<std::size_t>(std::ranges::size(_counts))};
    }

    /// @return The per-rank displacements, flattening the source first if needed.
    std::span<int const> mpi_displs() const {
        ensure_flattened();
        return {std::ranges::data(_displs), static_cast<std::size_t>(std::ranges::size(_displs))};
    }

    /// Displacements are always computed via exclusive_scan — monotonically non-decreasing.
    constexpr bool displs_monotonic() const {
        return true;
    }

    constexpr Counts const& counts() const& {
        return _counts;
    }
    constexpr Counts& counts() & {
        return _counts;
    }
    constexpr Counts&& counts() && {
        return std::move(_counts);
    }

    constexpr Displs const& displs() const& {
        return _displs;
    }
    constexpr Displs& displs() & {
        return _displs;
    }
    constexpr Displs&& displs() && {
        return std::move(_displs);
    }
};

template <typename S, typename F, typename C, typename D>
flatten_v_view(S&&, F&&, C&&, D&&)
    -> flatten_v_view<kamping::v2::all_t<S>, kamping::v2::all_t<F>, kamping::v2::all_t<C>, kamping::v2::all_t<D>>;

template <typename Source, typename FlatBuf, typename Counts, typename Displs, bool rb, bool rc, bool rd>
inline constexpr bool enable_borrowed_buffer<flatten_v_view<Source, FlatBuf, Counts, Displs, rb, rc, rd>> =
    enable_borrowed_buffer<FlatBuf> && enable_borrowed_buffer<Counts> && enable_borrowed_buffer<Displs>;

/// A flatten_v_view is flat: it never satisfies a structure concept, whatever its element type
/// looks like (see already_flat_buffer).
template <typename Source, typename FlatBuf, typename Counts, typename Displs, bool rb, bool rc, bool rd>
inline constexpr bool already_flat_buffer<flatten_v_view<Source, FlatBuf, Counts, Displs, rb, rc, rd>> = true;

} // namespace kamping::v2

namespace kamping::v2::views {

/// Flattens a structured send buffer: `per_rank | views::flatten_v()`.
///
/// The flat buffer, counts and displacements are allocated internally and resized as needed.
/// FlatTemplate/CountsContainer/DisplsContainer select the internal container types:
///   flatten_v<std::deque>()                       — custom flat buffer container
///   flatten_v<std::vector, std::deque<int>>()     — custom counts container
template <
    template <typename...> typename FlatTemplate = std::vector,
    typename CountsContainer                     = std::vector<int>,
    typename DisplsContainer                     = std::vector<int>>
constexpr auto flatten_v() {
    return kamping::v2::adaptor<0, decltype([](auto&& source) {
                                    using elem_t = payload_element_t<std::remove_cvref_t<decltype(source)>>;
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

/// Flattens into the caller-provided `flat_buf`, which must already be large enough; counts and displacements are
/// allocated internally.
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

/// Flattens into the caller-provided `flat_buf` and `counts`, which must already be large enough; displacements are
/// allocated internally.
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

/// Flattens into the caller-provided `flat_buf`, `counts` and `displs`, which must already be large enough.
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

/// Flattens into the caller-provided `flat_buf`, resizing it as needed; counts and displacements are allocated
/// internally.
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

/// Flattens into the caller-provided `flat_buf` and `counts`, resizing them as needed; displacements are allocated
/// internally.
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

/// Flattens into the caller-provided `flat_buf`, `counts` and `displs`, resizing them as needed.
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
