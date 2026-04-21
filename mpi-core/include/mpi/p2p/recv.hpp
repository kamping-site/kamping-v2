#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <
    recv_buffer                               RBuf,
    rank                                      Source = int,
    tag                                       Tag    = int,
    convertible_to_mpi_handle<MPI_Comm>       Comm   = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
void recv(
    RBuf&&      rbuf,
    Source      source = MPI_ANY_SOURCE,
    Tag         tag    = MPI_ANY_TAG,
    Comm const& comm   = MPI_COMM_WORLD,
    Status&&    status = MPI_STATUS_IGNORE
) {
    int err = MPI_Recv(
        ptr(rbuf),
        static_cast<int>(count(rbuf)),
        type(rbuf),
        to_rank(source),
        to_tag(tag),
        handle(comm),
        handle_ptr(status)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
