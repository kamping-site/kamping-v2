// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <numeric>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "kamping/kassert/kassert.hpp"
#include "mpi/buffer.hpp"

/// @file
/// Detail helpers shared by the grid (dstl/grid_alltoallv.hpp) and flat (dstl/flat_alltoallv.hpp)
/// all-to-all-v implementations: the buffer element-type trait, an exclusive-scan helper for
/// displacements, and the OpenMP `chunked_for` work-splitting kernel.

namespace dstl::detail {

/// The C++ element type behind a buffer: `ptr()` returns a typed pointer, e.g. `int const*` → `int`.
template <typename Buf>
using element_t = std::remove_cv_t<
    std::remove_pointer_t<decltype(mpi::experimental::ptr(std::declval<std::remove_cvref_t<Buf> const&>()))>>;

/// Exclusive prefix sum (displacements) of a counts array, in element units.
inline std::vector<int> exclusive_scan_int(std::span<int const> counts) {
    std::vector<int> displs(counts.size());
    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
    return displs;
}

/// Apply `body(lo, hi)` over contiguous chunks of `[0, n)` (TBB `blocked_range` style: each invocation
/// owns a disjoint slice). When `parallel` is set the range is split into one contiguous chunk per
/// OpenMP thread and the chunks run concurrently; otherwise `body` runs once over the whole range. The
/// `n*tid/nthreads` split tolerates `n` not divisible by the thread count: chunks stay contiguous, cover
/// `[0, n)` exactly, and differ in size by at most one element (empty chunks when threads outnumber
/// elements are fine — `body` then sees `lo == hi`). Parallel execution requires OpenMP, asserted here.
/// `body` must be safe to invoke concurrently across distinct slices.
template <typename F>
inline void chunked_for([[maybe_unused]] bool parallel, std::ptrdiff_t n, F&& body) {
#ifndef _OPENMP
    KAMPING_ASSERT(!parallel, "chunked_for: parallel execution requires OpenMP, which this build lacks.");
    body(std::ptrdiff_t{0}, n);
#else
    if (!parallel) {
        // Non-parallel: use the plain sequential path, not a one-thread OpenMP region.
        body(std::ptrdiff_t{0}, n);
    } else {
    #pragma omp parallel
        {
            std::ptrdiff_t const nthreads = omp_get_num_threads();
            std::ptrdiff_t const tid      = omp_get_thread_num();
            body(n * tid / nthreads, n * (tid + 1) / nthreads);
        }
    }
#endif
}

} // namespace dstl::detail
