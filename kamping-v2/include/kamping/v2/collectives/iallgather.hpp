// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/iresult.hpp"
#include "kamping/v2/result.hpp"
#include "mpi/collectives/iallgather.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {
/// Low-level overload: caller supplies an external request and manages its lifetime.
/// Returns result<SBuf, RBuf> immediately; the operation may still be in flight.
template <
    mpi::experimental::send_buffer                                SBuf,
    mpi::experimental::recv_buffer                                RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
auto iallgather(Request&& request, SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::allgather{}, res.send, res.recv, mpi::experimental::handle(comm));
    mpi::experimental::iallgather(res.send, res.recv, comm, std::forward<Request>(request));
    return res;
}

/// High-level overload: creates and owns the MPI_Request internally.
/// Returns iresult<SBuf, RBuf>; call wait() to block and retrieve the result.
///
/// `infer(comm_op::allgather{}, ...)` is called synchronously before `MPI_Iallgather`
/// is issued. For deferred recv buffers this sizes the buffer to comm_size * send_count
/// before the non-blocking call.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto iallgather(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) -> iresult<SBuf, RBuf> {
    iresult<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::allgather{}, res.send(), res.recv(), mpi::experimental::handle(comm));
    mpi::experimental::iallgather(res.send(), res.recv(), comm, res.mpi_native_handle_ptr());
    return res;
}
} // namespace kamping::v2
