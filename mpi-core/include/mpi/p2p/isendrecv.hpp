// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

#if MPI_VERSION >= 4

namespace mpi::experimental {

/// Starts sending `sbuf` to `dest` and receiving into `rbuf` from `source` (`MPI_Isendrecv`, MPI 4.0). The caller must
/// keep both buffers alive and complete `request`.
template <
    send_buffer                                SBuf,
    recv_buffer                                RBuf,
    rank                                       Dest,
    rank                                       Source,
    tag                                        SendTag,
    tag                                        RecvTag,
    convertible_to_mpi_handle<MPI_Comm>        Comm,
    convertible_to_mpi_handle_ptr<MPI_Request> Request>
void isendrecv(
    SBuf&&      sbuf,
    Dest        dest,
    SendTag     send_tag,
    RBuf&&      rbuf,
    Source      source,
    RecvTag     recv_tag,
    Comm const& comm,
    Request&&   request
) {
    int err = MPI_Isendrecv(
        mpi::experimental::ptr(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(send_tag),
        mpi::experimental::ptr(rbuf),
        static_cast<int>(mpi::experimental::count(rbuf)),
        mpi::experimental::type(rbuf),
        mpi::experimental::to_rank(source),
        mpi::experimental::to_tag(recv_tag),
        mpi::experimental::handle(comm),
        mpi::experimental::handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

} // namespace mpi::experimental

#endif // MPI_VERSION >= 4
