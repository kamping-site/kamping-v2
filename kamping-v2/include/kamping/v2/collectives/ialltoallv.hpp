// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/iresult.hpp"
#include "kamping/v2/result.hpp"
#include "mpi/collectives/ialltoallv.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {
/// Low-level overload: caller supplies an external request and manages its lifetime.
/// Returns result<SBuf, RBuf> immediately; the operation may still be in flight.
template <
    mpi::experimental::send_buffer_v                              SBuf,
    mpi::experimental::recv_buffer_v                              RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
auto ialltoallv(Request&& request, SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD)
    -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::alltoallv{}, res.send, res.recv, mpi::experimental::handle(comm));
    mpi::experimental::ialltoallv(res.send, res.recv, comm, std::forward<Request>(request));
    return res;
}

/// High-level overload: creates and owns the MPI_Request internally.
/// Returns iresult<SBuf, RBuf>; call wait() or test() to complete.
///
/// `infer(comm_op::alltoallv{}, ...)` runs a blocking MPI_Alltoall to exchange per-rank
/// counts before MPI_Ialltoallv is issued. The wire-level transfer is deferred.
///
/// If you need an `ialltoallv` that is guaranteed not to block on the caller's
/// thread, pass a recv buffers whose `infer()` overload is a no-op (for example,
/// a buffer not satisfying deferred_recv_buf_v).
template <
    mpi::experimental::send_buffer_v                        SBuf,
    mpi::experimental::recv_buffer_v                        RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>  Comm = MPI_Comm>
auto ialltoallv(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) -> iresult<SBuf, RBuf> {
    iresult<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::alltoallv{}, res.send(), res.recv(), mpi::experimental::handle(comm));
    mpi::experimental::ialltoallv(res.send(), res.recv(), comm, res.mpi_native_handle_ptr());
    return res;
}
} // namespace kamping::v2
