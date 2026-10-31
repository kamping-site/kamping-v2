// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Matches a message matching `source` and `tag` if one is available, without blocking (`MPI_Improbe`).
///
/// A matched message is removed from the matching queue and must be received through `message` with `mrecv()` or
/// `imrecv()`.
///
/// @param message Receives the handle of the matched message; untouched if no message is available.
/// @param status  Optional: filled with the status of the matched message. Defaults to `MPI_STATUS_IGNORE`.
/// @return `true` if a message was matched.
template <
    rank                                       Source,
    tag                                        Tag,
    convertible_to_mpi_handle_ptr<MPI_Message> Message,
    convertible_to_mpi_handle<MPI_Comm>        Comm,
    convertible_to_mpi_handle_ptr<MPI_Status>  Status = MPI_Status*>
bool improbe(Source source, Tag tag, Comm const& comm, Message&& message, Status&& status = MPI_STATUS_IGNORE) {
    int flag = 0;
    int err  = MPI_Improbe(
        mpi::experimental::to_rank(source),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        &flag,
        mpi::experimental::handle_ptr(message),
        mpi::experimental::handle_ptr(status)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return flag != 0;
}
} // namespace mpi::experimental
