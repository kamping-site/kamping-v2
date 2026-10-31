// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Receives the matched message `message` into `rbuf` (`MPI_Mrecv`); resets `message` to `MPI_MESSAGE_NULL`.
template <
    recv_buffer                                RBuf,
    convertible_to_mpi_handle_ptr<MPI_Message> Message = MPI_Message*,
    convertible_to_mpi_handle_ptr<MPI_Status>  Status  = MPI_Status*>
void mrecv(RBuf&& rbuf, Message&& message, Status&& status = MPI_STATUS_IGNORE) {
    int err = MPI_Mrecv(ptr(rbuf), static_cast<int>(count(rbuf)), type(rbuf), handle_ptr(message), handle_ptr(status));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
