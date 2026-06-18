// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#include "dstl/default_init_allocator.hpp"

/// @file
/// Per-thread counts/displacements setup for the `thread_multiple` grid alltoallv. These helpers are
/// only relevant to that execution policy: they split each phase's per-(rank, bucket) counts across the
/// OpenMP threads and pack each thread's contiguous send buffer, so the orchestration in
/// `route_phase_thread_multiple` (alltoallv.hpp) stays readable and the other policies are untouched.

namespace dstl::detail {

/// Per-subcomm-rank send/recv layout (counts + exclusive-scan displacements) of a phase exchange.
struct rank_layout {
    std::vector<int> counts; // one entry per subcommunicator rank
    std::vector<int> displs; // exclusive scan of `counts`
    int              total = 0;
};

/// Collapse a per-(rank, bucket) count array (rank-major, `next_subtree_size` buckets per rank) into a
/// per-rank `rank_layout`. Used for both the send side (from a thread's split of `dest_counts`) and the
/// recv side (from a thread's received per-bucket metadata) of a `thread_multiple` phase.
inline rank_layout rank_layout_from_meta(std::span<int const> meta, int subcomm_size, std::size_t next_subtree_size) {
    rank_layout lay;
    lay.counts.assign(static_cast<std::size_t>(subcomm_size), 0);
    for (int r = 0; r < subcomm_size; ++r) {
        int sum = 0;
        for (std::size_t b = 0; b < next_subtree_size; ++b) {
            sum += meta[static_cast<std::size_t>(r) * next_subtree_size + b];
        }
        lay.counts[static_cast<std::size_t>(r)] = sum;
    }
    lay.displs.resize(lay.counts.size());
    std::exclusive_scan(lay.counts.begin(), lay.counts.end(), lay.displs.begin(), 0);
    lay.total = subcomm_size > 0 ? lay.displs.back() + lay.counts.back() : 0;
    return lay;
}

/// Number of elements OpenMP thread `tid` of `nthreads` owns when a block of `count` elements is split
/// as evenly as possible (the low-index threads absorb the `count % nthreads` remainder). Mirrors the
/// multi-threaded all-to-all reference: `count/nthreads + (tid < count%nthreads)`.
inline int thread_split(int count, int tid, int nthreads) {
    return count / nthreads + (tid < count % nthreads ? 1 : 0);
}

/// Number of elements owned by threads `[0, tid)` under `thread_split` — i.e. thread `tid`'s start
/// offset inside the block. Equals `(count/nthreads)*tid + min(tid, count%nthreads)`.
inline int thread_split_prefix(int count, int tid, int nthreads) {
    int const base = count / nthreads;
    int const rem  = count % nthreads;
    return base * tid + std::min(tid, rem);
}

/// One thread's plan for sending its share of a `thread_multiple` phase: for every (dest rank, bucket)
/// slot, the element `count` it ships and the `read_off` to gather that slice from the shared data
/// array, plus the resulting per-rank send layout into the thread's packed buffer.
struct thread_slot_plan {
    std::vector<int> count;    // [slot] this thread's element count for the slot
    std::vector<int> read_off; // [slot] read offset of the slice inside the shared (data) array
    rank_layout      ranks;    // per-rank counts/displs into the packed send buffer
};

/// Build thread `tid`'s send plan from the phase's `dest_counts` (per-(rank, bucket) counts) and
/// `slot_displs` (start of each slot's block in the shared data array). Each slot is split across the
/// `nthreads` threads with `thread_split`; the thread's slice of slot `s` is the contiguous range of the
/// shared array starting at `slot_displs[s] + thread_split_prefix(...)`.
inline thread_slot_plan plan_thread_send(
    std::span<int const> dest_counts,
    std::span<int const> slot_displs,
    int                  subcomm_size,
    std::size_t          next_subtree_size,
    int                  tid,
    int                  nthreads
) {
    auto const       subtree_size = dest_counts.size();
    thread_slot_plan plan;
    plan.count.assign(subtree_size, 0);
    plan.read_off.assign(subtree_size, 0);
    for (std::size_t s = 0; s < subtree_size; ++s) {
        plan.count[s]    = thread_split(dest_counts[s], tid, nthreads);
        plan.read_off[s] = slot_displs[s] + thread_split_prefix(dest_counts[s], tid, nthreads);
    }
    plan.ranks = rank_layout_from_meta(plan.count, subcomm_size, next_subtree_size);
    return plan;
}

/// Pack thread `tid`'s slice of every slot from the shared `src` array into `out`, in slot order
/// (rank-major / bucket-minor). Because the slots of a given rank are consecutive, rank blocks land
/// contiguously in `out`, matching `plan.ranks` — so `out` is ready for one MPI_Alltoallv.
template <typename T>
void pack_by_plan(thread_slot_plan const& plan, T const* src, uninit_vector<T>& out) {
    std::size_t write = 0;
    for (std::size_t s = 0; s < plan.count.size(); ++s) {
        int const cnt = plan.count[s];
        if (cnt > 0) {
            std::copy_n(src + plan.read_off[s], cnt, out.begin() + static_cast<std::ptrdiff_t>(write));
            write += static_cast<std::size_t>(cnt);
        }
    }
}

} // namespace dstl::detail
