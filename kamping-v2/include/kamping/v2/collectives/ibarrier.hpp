// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "kamping/v2/request.hpp"
#include "mpi/collectives/ibarrier.hpp"

namespace kamping::v2 {
using mpi::experimental::ibarrier; // ibarrier(comm, request): with a caller-supplied request

/// @brief Starts a non-blocking barrier on `comm` (`MPI_Ibarrier`).
///
/// Since a barrier has no buffer, this returns a plain @ref request rather than an `iresult`. It completes once all
/// ranks of `comm` have started the barrier and must be completed with `wait()` or a successful `test()`.
/// @return The request of the barrier.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] request ibarrier(Comm const& comm) {
    request req;
    mpi::experimental::ibarrier(comm, req);
    return req;
}
} // namespace kamping::v2
