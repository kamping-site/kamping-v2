// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

/// Sends `sbuf` to `dest` and receives into `rbuf` from `source` in one call (`MPI_Sendrecv`).
///
/// @param status Filled with the status of the received message; pass `MPI_STATUS_IGNORE` if not needed.
template <
    send_buffer                               SBuf,
    recv_buffer                               RBuf,
    rank                                      Dest,
    rank                                      Source,
    tag                                       SendTag,
    tag                                       RecvTag,
    convertible_to_mpi_handle<MPI_Comm>       Comm,
    convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
void sendrecv(
    SBuf&&      sbuf,
    Dest        dest,
    SendTag     send_tag,
    RBuf&&      rbuf,
    Source      source,
    RecvTag     recv_tag,
    Comm const& comm,
    Status&&    status
) {
    int err = MPI_Sendrecv(
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
        mpi::experimental::handle_ptr(status)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

} // namespace mpi::experimental
