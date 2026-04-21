#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <send_recv_buffer SRBuf, rank Root = int, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void bcast(SRBuf&& send_recv_buf, Root root = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Bcast(
        ptr(send_recv_buf),
        static_cast<int>(count(send_recv_buf)),
        type(send_recv_buf),
        to_rank(root),
        handle(comm)

    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
