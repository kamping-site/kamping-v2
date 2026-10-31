// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

/// Blocks until a message matching `source` and `tag` is available, without receiving it (`MPI_Probe`).
///
/// @param status Optional: filled with the status of the message. Defaults to `MPI_STATUS_IGNORE`.
template <
    rank                                      Source,
    tag                                       Tag,
    convertible_to_mpi_handle<MPI_Comm>       Comm,
    convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
void probe(Source source, Tag tag, Comm const& comm, Status&& status = MPI_STATUS_IGNORE) {
    int err = MPI_Probe(
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
