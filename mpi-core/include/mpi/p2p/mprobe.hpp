// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

/// Blocks until a message matching `source` and `tag` arrives and matches it (`MPI_Mprobe`).
///
/// The message is removed from the matching queue and must be received through `message` with `mrecv()` or `imrecv()`.
template <
    rank                                       Source  = int,
    tag                                        Tag     = int,
    convertible_to_mpi_handle_ptr<MPI_Message> Message = MPI_Message*,
    convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Status>  Status  = MPI_Status*>
void mprobe(Source source, Tag tag, Comm const& comm, Message&& message, Status&& status = MPI_STATUS_IGNORE) {
    int err = MPI_Mprobe(to_rank(source), to_tag(tag), handle(comm), handle_ptr(message), handle_ptr(status));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
