#pragma once

#include <mpi.h>

#include "kamping/kassert/kassert.hpp"
#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <send_buffer SBuf, recv_buffer RBuf, rank Root, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void gather(SBuf&& sbuf, RBuf&& rbuf, Root root, Comm const& comm) {
    int comm_size = 0;
    int comm_rank = 0;
    MPI_Comm_size(handle(comm), &comm_size);
    MPI_Comm_rank(handle(comm), &comm_rank);
    KAMPING_ASSERT(
        to_rank(root) != comm_rank || static_cast<int>(count(rbuf)) % comm_size == 0,
        "on root: recv buffer size must be divisible by comm size"
    );
    int err = MPI_Gather(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        ptr(rbuf),
        static_cast<int>(count(rbuf)) / comm_size,
        type(rbuf),
        to_rank(root),
        handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
