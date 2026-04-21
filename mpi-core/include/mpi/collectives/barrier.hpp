#pragma once

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void barrier(Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Barrier(handle(comm));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
