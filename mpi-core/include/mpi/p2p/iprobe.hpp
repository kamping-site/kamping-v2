// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

/// Checks whether a message matching `source` and `tag` is available, without blocking or receiving it
/// (`MPI_Iprobe`).
///
/// @param status Optional: filled with the status of the message if one is available. Defaults to
///               `MPI_STATUS_IGNORE`.
/// @return `true` if a matching message is available.
template <
    rank                                      Source = int,
    tag                                       Tag    = int,
    convertible_to_mpi_handle<MPI_Comm>       Comm   = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
bool iprobe(Source source, Tag tag, Comm const& comm, Status&& status = MPI_STATUS_IGNORE) {
    int flag = 0;
    int err  = MPI_Iprobe(to_rank(source), to_tag(tag), handle(comm), &flag, handle_ptr(status));
    if (err != MPI_SUCCESS) {
        throw experimental::mpi_error(err);
    }
    return flag != 0;
}
} // namespace mpi::experimental
