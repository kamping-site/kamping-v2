// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <format>
#include <span>
#include <string>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "dstl/grid_comm.hpp"
#include "dstl/tags.hpp"
#include "kamping/v2/views.hpp"
#include "mpi/collectives/allgather.hpp"
#include "mpi/comm.hpp"

/// @file
/// dstl visualization helpers — human-readable dumps of a communicator's thread layout and of a grid
/// communicator's dimension-by-dimension rank breakdown.
///
/// Two entry points, with deliberately different communication contracts:
///   * `dstl::as_string(...)` is **collective**: it gathers each rank's thread count across the
///     communicator (via `mpi::experimental::allgather`), so the returned string carries the full
///     per-rank picture on every rank.
///   * the `std::formatter<dstl::grid_comm<Exec>>` specialization (`std::format("{}", grid)`) is
///     **local**: it renders only the grid structure, which is a closed-form function of the dims and
///     therefore computable without any communication. It carries no per-rank thread counts — use
///     `as_string` for those.

namespace dstl {

/// @return The number of threads the calling rank actually runs with in an OpenMP parallel region,
/// measured by entering one and reading `omp_get_num_threads()` from inside it; `1` in a build without
/// OpenMP. This reflects the real team size the runtime grants (e.g. after dynamic adjustment or
/// affinity limits), unlike `omp_get_max_threads()`, which is only the configured upper bound.
[[nodiscard]] inline int local_thread_count() noexcept {
#ifdef _OPENMP
    int n = 1;
    #pragma omp parallel
    {
    // One thread records the team size; the single's implicit barrier publishes it before the
    // region (and its closing barrier) ends, so the read below is race-free.
    #pragma omp single
        n = omp_get_num_threads();
    }
    return n;
#else
    return 1;
#endif
}

/// Collectively gather every rank's `local_thread_count()` over `comm`.
/// @return A vector of size `comm.size()` indexed by rank, identical on all ranks.
/// @note Collective over `comm` — every rank must call it.
[[nodiscard]] inline std::vector<int> gather_thread_counts(mpi::experimental::comm_view comm) {
    int const        me = local_thread_count();
    auto const       p  = static_cast<std::size_t>(comm.size());
    std::vector<int> counts(p);
    mpi::experimental::allgather(kamping::v2::views::ref_single(me), counts, comm);
    return counts;
}

namespace detail {

/// Decimal width needed to print the largest rank in `[0, p)` (so a column of rank numbers aligns).
inline int rank_width(std::size_t p) noexcept {
    int width = 1;
    for (std::size_t hi = (p > 0 ? p - 1 : 0); hi >= 10; hi /= 10) {
        ++width;
    }
    return width;
}

template <execution_policy Exec>
inline std::string render_grid(grid_comm<Exec> const& grid, std::span<int const> threads) {
    auto const        p = static_cast<std::size_t>(grid.size());
    std::size_t const k = grid.num_dims();

    std::string dims_s;
    std::string strides_s;
    for (std::size_t i = 0; i < k; ++i) {
        dims_s += std::format("{}{}", i ? ", " : "", grid.dim_size(i));
        strides_s += std::format("{}{}", i ? ", " : "", grid.stride(i));
    }
    std::string out = std::format("grid_comm: p={}, dims=[{}], strides=[{}]\n", p, dims_s, strides_s);

    int const width    = rank_width(p);
    auto      fmt_rank = [&](std::size_t r) {
        return threads.empty() ? std::format("{:>{}}", r, width) : std::format("{:>{}}({})", r, width, threads[r]);
    };

    for (std::size_t i = 0; i < k; ++i) {
        std::size_t const s        = grid.dim_size(i);
        std::size_t const stride   = grid.stride(i);
        std::size_t const n_fibers = (s > 0 ? p / s : 0);
        char const* const kind     = (k == 1)       ? "single dimension"
                                     : (i == 0)     ? "most remote, inter-group"
                                     : (i + 1 == k) ? "most local, intra-group"
                                                    : "intermediate";
        out += std::format(" dim {} (size {}, stride {}): {} subcomm(s) [{}]\n", i, s, stride, n_fibers, kind);
        for (std::size_t base = 0; base < p; ++base) {
            if ((base / stride) % s != 0) {
                continue; // not a fiber base: coordinate i is non-zero here
            }
            std::string members;
            for (std::size_t j = 0; j < s; ++j) {
                members += std::format("{}{}", j ? ", " : "", fmt_rank(base + j * stride));
            }
            out += std::format("   {{ {} }}\n", members);
        }
    }
    return out;
}

} // namespace detail

/// Collectively build a per-rank thread-count table for `comm` (e.g. `rank 3: 8 thread(s)`).
/// @note Collective over `comm` — every rank must call it; the returned string is identical on all ranks.
[[nodiscard]] inline std::string as_string(mpi::experimental::comm_view comm) {
    auto const threads = gather_thread_counts(comm);
    auto const p       = threads.size();
    int        total   = 0;
    for (int const t: threads) {
        total += t;
    }
    std::string out   = std::format("comm: {} rank(s), {} thread(s) total\n", p, total);
    int const   width = detail::rank_width(p);
    for (std::size_t r = 0; r < p; ++r) {
        out += std::format("  rank {:>{}}: {} thread(s)\n", r, width, threads[r]);
    }
    return out;
}

/// Collectively build the grid breakdown for `grid`, annotated with each rank's thread count
/// (`r(threads)`). Gathers thread counts over the global communicator, then renders the structure.
/// @note Collective over `grid.global()` — every rank must call it; the result is identical on all ranks.
/// For a communication-free structural dump use `std::format("{}", grid)` instead.
template <execution_policy Exec>
[[nodiscard]] inline std::string as_string(grid_comm<Exec> const& grid) {
    auto const threads = gather_thread_counts(grid.global());
    return detail::render_grid(grid, threads);
}

} // namespace dstl

/// Local (communication-free) `std::format` support for `grid_comm`: renders the dimension-by-dimension
/// rank breakdown. No per-rank thread counts (those need a gather — see `dstl::as_string`).
template <dstl::execution_policy Exec>
struct std::formatter<dstl::grid_comm<Exec>, char> {
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(dstl::grid_comm<Exec> const& grid, FormatContext& ctx) const {
        return std::format_to(ctx.out(), "{}", dstl::detail::render_grid(grid, std::span<int const>{}));
    }
};
