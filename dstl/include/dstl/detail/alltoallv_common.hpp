// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <cstddef>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "kamping/kassert/kassert.hpp"

/// @file
/// The OpenMP work-splitting kernels shared by the dstl collectives — `chunked_for` (split-by-thread-id
/// inside one parallel region) and `chunking` (a fixed split-by-chunk-index partition). Used by the grid
/// (dstl/grid_alltoallv.hpp), flat (dstl/flat_alltoallv.hpp), and request_reply bucketing.

namespace dstl::detail {

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

/// A fixed partition of `[0, n)` into `count()` contiguous chunks, addressed by chunk *index*. Unlike
/// `chunked_for` (which splits by thread id inside a single parallel region), the bounds here are a pure
/// function of the chunk index, so several separate `for_each_chunk` passes over the same `chunking`
/// (e.g. a count pass then a write pass) agree on identical chunk boundaries no matter how OpenMP
/// schedules the iterations — key all per-chunk state by the chunk index `c`. When `parallel` is set
/// there is one chunk per OpenMP thread, capped at `n` so chunks never outnumber elements; otherwise a
/// single chunk spans the whole range. The `n*c/count` split keeps chunks contiguous, covers `[0, n)`
/// exactly, and balances sizes to within one element (empty chunks — when threads outnumber elements —
/// are fine). Parallel use requires OpenMP, asserted in the ctor (as `chunked_for`).
struct chunking {
    std::ptrdiff_t n        = 0;
    int            nchunks  = 1;
    bool           parallel = false;

    chunking() = default;

    chunking(bool parallel_, std::ptrdiff_t n_) : n{n_}, parallel{parallel_} {
#ifdef _OPENMP
        if (parallel) {
            nchunks = std::clamp(omp_get_max_threads(), 1, static_cast<int>(std::max<std::ptrdiff_t>(n, 1)));
        }
#else
        KAMPING_ASSERT(!parallel, "chunking: parallel execution requires OpenMP, which this build lacks.");
#endif
    }

    /// Number of chunks (`>= 1`).
    [[nodiscard]] int count() const noexcept {
        return nchunks;
    }
    /// First index of chunk `c` (in `[0, count())`).
    [[nodiscard]] std::ptrdiff_t begin(int c) const noexcept {
        return n * c / nchunks;
    }
    /// One-past-the-last index of chunk `c`.
    [[nodiscard]] std::ptrdiff_t end(int c) const noexcept {
        return n * (c + 1) / nchunks;
    }

    /// Invoke `body(c, begin(c), end(c))` for every chunk `c` — the chunk-indexed analogue of
    /// `chunked_for`. The chunks run concurrently across OpenMP threads when this `chunking` was built
    /// `parallel` (one chunk per thread via `schedule(static)`); otherwise they run on a plain sequential
    /// path (no OpenMP region). The chunk index is passed so `body` can key its per-chunk state by `c`
    /// (see the class note); `body` must be safe to invoke concurrently across distinct chunks.
    template <typename F>
    void for_each_chunk(F&& body) const {
#ifdef _OPENMP
        if (parallel) {
    #pragma omp parallel for schedule(static)
            for (int c = 0; c < nchunks; ++c) {
                body(c, begin(c), end(c));
            }
            return;
        }
#endif
        // Sequential path: not a one-thread OpenMP region.
        for (int c = 0; c < nchunks; ++c) {
            body(c, begin(c), end(c));
        }
    }
};

} // namespace dstl::detail
