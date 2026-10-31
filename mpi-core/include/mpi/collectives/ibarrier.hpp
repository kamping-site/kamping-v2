// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Starts a non-blocking barrier on `comm` (`MPI_Ibarrier`).
///
/// The operation completes once all ranks of `comm` have started the barrier.
///
/// @param request Receives the request of the barrier; the caller must complete it.
template <convertible_to_mpi_handle<MPI_Comm> Comm, convertible_to_mpi_handle_ptr<MPI_Request> Request>
void ibarrier(Comm const& comm, Request&& request) {
    int err = MPI_Ibarrier(mpi::experimental::handle(comm), mpi::experimental::handle_ptr(request));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
