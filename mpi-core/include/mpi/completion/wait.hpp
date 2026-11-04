// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

/// Blocks until the operation associated with `request` completes (`MPI_Wait`).
///
/// On return, `request` is set to `MPI_REQUEST_NULL` (or marked inactive, for persistent requests).
/// Waiting on `MPI_REQUEST_NULL` returns immediately with an empty status.
///
/// @param request Anything convertible to an `MPI_Request*` (raw `MPI_Request`, `request_view`, …).
/// @param status  Optional status out-parameter; omit to use `MPI_STATUS_IGNORE`.
template <
    convertible_to_mpi_handle_ptr<MPI_Request> Request,
    convertible_to_mpi_handle_ptr<MPI_Status>  Status = MPI_Status*>
void wait(Request&& request, Status&& status = MPI_STATUS_IGNORE) {
    int err = MPI_Wait(mpi::experimental::handle_ptr(request), mpi::experimental::handle_ptr(status));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// Checks whether the operation associated with `request` has completed (`MPI_Test`).
///
/// If it has, `request` is set to `MPI_REQUEST_NULL` (or marked inactive, for persistent requests)
/// and `status` is filled in. Testing `MPI_REQUEST_NULL` returns `true`.
///
/// @param request Anything convertible to an `MPI_Request*` (raw `MPI_Request`, `request_view`, …).
/// @param status  Optional status out-parameter; omit to use `MPI_STATUS_IGNORE`.
/// @return `true` if the operation has completed.
template <
    convertible_to_mpi_handle_ptr<MPI_Request> Request,
    convertible_to_mpi_handle_ptr<MPI_Status>  Status = MPI_Status*>
bool test(Request&& request, Status&& status = MPI_STATUS_IGNORE) {
    int flag = 0;
    int err  = MPI_Test(mpi::experimental::handle_ptr(request), &flag, mpi::experimental::handle_ptr(status));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return flag != 0;
}

} // namespace mpi::experimental
