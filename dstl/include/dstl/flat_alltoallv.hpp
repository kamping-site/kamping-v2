// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <mpi.h>

#include "dstl/thread_multiple_comm.hpp"
#include "kamping/kassert/kassert.hpp"
#include "kamping/v2/result.hpp"
#include "kamping/v2/views.hpp"
#include "mpi/buffer.hpp"
#include "mpi/collectives/alltoall.hpp"
#include "mpi/collectives/alltoallv.hpp"
#include "mpi/comm.hpp"

/// @file
/// dstl::alltoallv over a dstl::thread_multiple_comm — a single-exchange (flat) all-to-all-v that parallelizes
/// the MPI exchange itself across OpenMP threads (the MultiThreadedAlltoAll algorithm): each
/// destination block is split across `T` threads and `T` concurrent count+data `MPI_Alltoallv` calls
/// run on the `T` duplicated communicators owned by the `thread_multiple_comm`. The per-thread recv
/// displacements lay each source's contributions out contiguously in rank-then-thread order, so the
/// result is byte-identical to a flat `MPI_Alltoallv` (no recv-ordering tag is needed).
///
/// The `seq` / `par` execution policies need no overload here: their exchange is a single
/// `MPI_Alltoallv`, i.e. exactly `kamping::v2::alltoallv`. The `par` policy's parallel send-buffer
/// pack is a separate helper (see DSTL-Flat-Alltoallv-Design.md §0/§9).

namespace dstl {

/// Recv buffer accepted by the flat alltoallv: a resizable, contiguous range whose element type
/// matches the send buffer's. The total is computed locally and written as one contiguous block.
template <typename RBuf>
concept flat_recv_buffer =
    mpi::experimental::send_buffer_v<RBuf> && std::copyable<std::ranges::range_value_t<std::remove_cvref_t<RBuf>>>;

namespace detail {

template <typename F>
inline void for_each_thread([[maybe_unused]] int nthreads, F&& body) {
#ifndef _OPENMP
    for (int tid = 0; tid < nthreads; ++tid)
        body(tid);
#else
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int tid = 0; tid < nthreads; ++tid) {
        body(tid);
    }
#endif
}

} // namespace detail

/// Drop-in flat all-to-all-v dispatched on a `dstl::thread_multiple_comm`. The call site matches the flat
/// KaMPIng v2 alltoallv; only the communicator changes from a flat comm to a `thread_multiple_comm`. The send
/// buffer is a standard variadic `send_buffer_v` (`data | with_counts | with_displs`); the recv
/// buffer is any resizable contiguous range (e.g. a plain `std::vector<T>`).
///
/// Implements the MultiThreadedAlltoAll algorithm over the duplicated communicators owned by the
/// `thread_multiple_comm`: each destination block is split across the `T` threads and `T` concurrent
/// count+data `MPI_Alltoallv` calls run, one per duplicated communicator. The local bookkeeping is
/// O(T*p) and cheap, so it is left serial.
///
/// Unlike the grid wrapper there is no recv-ordering tag: the per-thread recv displacements already
/// reassemble each source's block in rank order, so the result is always flat-identical.
template <mpi::experimental::send_buffer_v SBuf, flat_recv_buffer RBuf>
auto alltoallv(SBuf&& sbuf, RBuf&& rbuf, thread_multiple_comm const& fc) -> kamping::v2::result<SBuf, RBuf> {
    namespace views = kamping::v2::views;

    auto const        thread_comms    = fc.thread_comms();
    std::size_t const p               = static_cast<std::size_t>(fc.size());
    std::size_t const nthreads        = thread_comms.size();
    int const         nthreads_signed = static_cast<int>(nthreads);
    auto const        send_counts     = mpi::experimental::counts(sbuf);
    auto const        send_displs     = mpi::experimental::displs(sbuf);

    // Split each destination's send block contiguously across the T threads. Thread t receives the
    // remainder-adjusted slice [t*base + min(t, rem), …); concatenating the slices in thread order
    // reproduces the destination's original block, which is what makes the result flat-identical.
    std::vector<std::vector<int>> thread_send_counts(nthreads, std::vector<int>(p, 0));
    std::vector<std::vector<int>> thread_send_displs(nthreads, std::vector<int>(p, 0));
    for (std::size_t dest = 0; dest < p; ++dest) {
        int const cnt  = send_counts[dest];
        int const base = cnt / nthreads_signed;
        int const rem  = cnt % nthreads_signed;
        int const off  = send_displs[dest];
        for (std::size_t t = 0; t < nthreads; ++t) {
            int const tid               = static_cast<int>(t);
            int const extra             = tid < rem ? tid : rem; // min(t, rem)
            thread_send_counts[t][dest] = base + (tid < rem ? 1 : 0);
            thread_send_displs[t][dest] = off + tid * base + extra;
        }
    }

    // Each thread negotiates its own per-source recv counts on its own communicator.
    std::vector<std::vector<int>> thread_recv_counts(nthreads, std::vector<int>(p, 0));
    detail::for_each_thread(nthreads_signed, [&](int tid) {
        auto const t = static_cast<std::size_t>(tid);
        mpi::experimental::alltoall(thread_send_counts[t], thread_recv_counts[t], thread_comms[t]);
    });

    // Aggregate the per-source totals and resize the recv buffer.
    if constexpr (kamping::v2::deferred_recv_buf_v<std::remove_cvref_t<RBuf>>) {
        rbuf.set_comm_size(static_cast<int>(p));
        for (std::size_t t = 0; t < nthreads; ++t) {
            for (std::size_t s = 0; s < p; ++s) {
                mpi::experimental::counts(rbuf)[s] += thread_recv_counts[t][s];
            }
        }
        rbuf.commit_counts();
    }
    // Per-thread recv displacements: within each source's block the threads' contributions are laid
    // out in thread order, so the source's block ends up contiguous and in its original element order.
    std::vector<std::vector<int>> thread_recv_displs(nthreads, std::vector<int>(p, 0));
    for (std::size_t s = 0; s < p; ++s) {
        int cursor = mpi::experimental::displs(rbuf)[s];
        for (std::size_t t = 0; t < nthreads; ++t) {
            thread_recv_displs[t][s] = cursor;
            cursor += thread_recv_counts[t][s];
        }
    }

    mpi::experimental::ptr(rbuf); // triggers resizing ... if the buffer is resizable

    // Concurrent data exchange: each thread ships its slices into disjoint regions of rbuf.
    detail::for_each_thread(nthreads_signed, [&](int tid) {
        auto const t = static_cast<std::size_t>(tid);
        mpi::experimental::alltoallv(
            sbuf | views::with_counts(thread_send_counts[t]) | views::with_displs(thread_send_displs[t]),
            rbuf | views::with_counts(thread_recv_counts[t]) | views::with_displs(thread_recv_displs[t]),
            thread_comms[t]
        );
    });

    return kamping::v2::result<SBuf, RBuf>{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
}

} // namespace dstl
