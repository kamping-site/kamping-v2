// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <kamping/types/reduce_ops.hpp>
#include <mpi.h>

#include "dstl/tags.hpp"
#include "kamping/kassert/kassert.hpp"
#include "kamping/v2/collectives/allreduce.hpp"
#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "mpi/comm.hpp"
#include "mpi/environment.hpp"

/// @file
/// dstl::thread_multiple_comm — a single-exchange (flat) all-to-all-v communicator that parallelizes the MPI
/// exchange *itself* across OpenMP threads under MPI_THREAD_MULTIPLE: each destination block is split
/// across `T` threads and `T` concurrent MPI_Alltoallv calls are issued, one per duplicated
/// communicator (the MultiThreadedAlltoAll algorithm).
///
/// The `seq` and `par` execution policies need no such wrapper: their exchange is a single
/// MPI_Alltoallv, identical to `kamping::v2::alltoallv` — so callers use that directly. The `par`
/// policy's only intra-rank win is parallelizing the *send-buffer pack*, exposed as a separate helper
/// (see DSTL-Flat-Alltoallv-Design.md §0/§9), not as a thread_multiple_comm or alltoallv overload.
///

namespace dstl {

/// Number of OpenMP threads the thread_multiple exchange parallelizes over (1 without OpenMP).
[[nodiscard]] inline int flat_max_threads() noexcept {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/// (Flat) communicator for the thread_multiple model: it owns
/// `t = flat_max_threads()` duplicated communicators so that `t` concurrent MPI_Alltoallv calls can
/// run, one per thread.
class thread_multiple_comm : public mpi::experimental::comm_accessors<thread_multiple_comm> {
public:
    using execution_policy_type = execution_policy::par_comm;

    /// Borrow `global` and duplicate `t` communicators (collective). Asserts that the runtime
    /// provides MPI_THREAD_MULTIPLE.
    ///
    /// `t` must be identical on every rank: both MPI_Comm_dup here and the per-thread MPI_Alltoallv
    /// later are collective, so thread `t` on every rank has to participate. `flat_max_threads()`
    /// (i.e. `omp_get_max_threads()`) can legitimately differ across ranks under CPU affinity or
    /// oversubscription, so we agree on the maximum via an MPI_Allreduce before duplicating.
    explicit thread_multiple_comm(mpi::experimental::comm_view global) : _global(global) {
        KAMPING_ASSERT(
            mpi::experimental::environment::thread_level() >= mpi::experimental::ThreadLevel::multiple,
            "thread_multiple_comm<thread_multiple>: the MPI runtime does not provide MPI_THREAD_MULTIPLE"
        );
        int t = flat_max_threads();
        kamping::v2::allreduce(kamping::v2::inplace, kamping::v2::views::ref_single(t), kamping::ops::max<>{}, _global);
        _thread_comms.reserve(static_cast<std::size_t>(t));
        for (int i = 0; i < t; ++i) {
            _thread_comms.push_back(_global.dup());
        }
    }

    thread_multiple_comm(thread_multiple_comm const&)            = delete; // owns the dup comms
    thread_multiple_comm& operator=(thread_multiple_comm const&) = delete;
    thread_multiple_comm(thread_multiple_comm&&)                 = default;
    thread_multiple_comm& operator=(thread_multiple_comm&&)      = default;
    ~thread_multiple_comm()                                      = default;

    /// @return The underlying global `MPI_Comm`. This is the `mpi_handle()` dispatch point: it makes
    /// `thread_multiple_comm` usable directly in normal communication contexts and is the handle the
    /// inherited `rank()` / `size()` / `group()` / `dup()` / `split()` (from `comm_accessors`) operate on.
    [[nodiscard]] MPI_Comm mpi_handle() const noexcept {
        return _global.mpi_handle();
    }

    /// @return A view of the underlying global communicator.
    [[nodiscard]] mpi::experimental::comm_view global() const noexcept {
        return _global;
    }

    /// @return The `t` per-thread communicators used by the concurrent exchange.
    [[nodiscard]] std::span<mpi::experimental::comm const> thread_comms() const noexcept {
        return {_thread_comms.data(), _thread_comms.size()};
    }

private:
    mpi::experimental::comm_view         _global;
    std::vector<mpi::experimental::comm> _thread_comms;
};

} // namespace dstl
