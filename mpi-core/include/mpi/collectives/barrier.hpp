// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Blocks until all ranks of `comm` have called it (`MPI_Barrier`).
template <convertible_to_mpi_handle<MPI_Comm> Comm>
void barrier(Comm const& comm) {
    int err = MPI_Barrier(mpi::experimental::handle(comm));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
