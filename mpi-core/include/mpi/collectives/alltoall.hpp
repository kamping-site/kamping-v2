#pragma once

#include <mpi.h>

#include "kamping/kassert/kassert.hpp"
#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <send_buffer SBuf, recv_buffer RBuf, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void alltoall(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) {
    int comm_size = 0;
    MPI_Comm_size(handle(comm), &comm_size);
    using scount_t = decltype(count(sbuf));
    using rcount_t = decltype(count(rbuf));
    KAMPING_ASSERT(
        count(sbuf) % static_cast<scount_t>(comm_size) == scount_t{0},
        "send buffer size must be divisible by comm size"
    );
    KAMPING_ASSERT(
        count(rbuf) % static_cast<rcount_t>(comm_size) == rcount_t{0},
        "recv buffer size must be divisible by comm size"
    );
    int err = MPI_Alltoall(
        ptr(sbuf),
        static_cast<int>(count(sbuf)) / comm_size,
        type(sbuf),
        ptr(rbuf),
        static_cast<int>(count(rbuf)) / comm_size,
        type(rbuf),
        handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
