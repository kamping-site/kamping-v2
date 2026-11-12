// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

/// Sends the contents of `send_recv_buf` to `dest` and replaces them with a message received from `source`, in one
/// call (`MPI_Sendrecv_replace`).
///
/// The same count and datatype are used for both directions: `count(send_recv_buf)` elements are sent, and the
/// received message must not be longer.
/// @param status Optional: filled with the status of the received message. Defaults to `MPI_STATUS_IGNORE`.
template <
    recv_buffer                               SRBuf,
    rank                                      Dest,
    rank                                      Source,
    tag                                       SendTag,
    tag                                       RecvTag,
    convertible_to_mpi_handle<MPI_Comm>       Comm,
    convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
void sendrecv_replace(
    SRBuf&&     send_recv_buf,
    Dest        dest,
    SendTag     send_tag,
    Source      source,
    RecvTag     recv_tag,
    Comm const& comm,
    Status&&    status = MPI_STATUS_IGNORE
) {
    int err = MPI_Sendrecv_replace(
        mpi::experimental::ptr(send_recv_buf),
        static_cast<int>(mpi::experimental::count(send_recv_buf)),
        mpi::experimental::type(send_recv_buf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(send_tag),
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
