// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Receives a message from `source` with `tag` into `rbuf` (`MPI_Recv`). `rbuf` must be large enough for the message.
///
/// @param status Optional: filled with the status of the received message. Defaults to `MPI_STATUS_IGNORE`.
template <
    recv_buffer                               RBuf,
    rank                                      Source,
    tag                                       Tag,
    convertible_to_mpi_handle<MPI_Comm>       Comm,
    convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
void recv(RBuf&& rbuf, Source source, Tag tag, Comm const& comm, Status&& status = MPI_STATUS_IGNORE) {
    int err = MPI_Recv(
        mpi::experimental::ptr(rbuf),
        static_cast<int>(mpi::experimental::count(rbuf)),
        mpi::experimental::type(rbuf),
        mpi::experimental::to_rank(source),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        mpi::experimental::handle_ptr(status)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
