#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

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
    requires std::ranges::forward_range<Source> && std::ranges::sized_range<Source>
             && std::ranges::input_range<std::ranges::range_value_t<Source>>
             && std::ranges::sized_range<std::ranges::range_value_t<Source>>
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

    void ensure_flattened() const {
        if (!needs_flatten_)
            return;

        auto const num_ranks = std::ranges::size(source_);

        // Resize and fill counts
        if constexpr (resize_counts) {
            kamping::v2::resize_for_receive(counts_, static_cast<std::ptrdiff_t>(num_ranks));
        }
        auto*          counts_ptr = std::ranges::data(counts_);
        std::ptrdiff_t total_size = 0;
        std::size_t    idx        = 0;
        for (auto&& inner: source_) {
            auto const s      = static_cast<int>(std::ranges::size(inner));
            counts_ptr[idx++] = s;
            total_size += s;
        }

        // Resize and compute displacements
        if constexpr (resize_displs) {
            kamping::v2::resize_for_receive(displs_, static_cast<std::ptrdiff_t>(num_ranks));
        }
        std::exclusive_scan(counts_ptr, counts_ptr + num_ranks, std::ranges::data(displs_), 0);

        // Resize and copy data into flat buffer
        if constexpr (resize_buf) {
            kamping::v2::resize_for_receive(flat_buf_, total_size);
        }
        using elem_t = std::ranges::range_value_t<std::ranges::range_value_t<Source>>;
        elem_t* dest = std::ranges::data(flat_buf_);
        for (auto&& inner: source_) {
            dest = std::copy(std::ranges::begin(inner), std::ranges::end(inner), dest);
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
                                    using Source  = std::remove_cvref_t<decltype(source)>;
                                    using inner_t = std::ranges::range_value_t<Source>;
                                    using elem_t  = std::ranges::range_value_t<inner_t>;
                                    using S       = kamping::v2::all_t<decltype(source)>;
                                    using F       = kamping::v2::owning_view<FlatTemplate<elem_t>>;
                                    using C       = kamping::v2::owning_view<CountsContainer>;
                                    using D       = kamping::v2::owning_view<DisplsContainer>;
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
