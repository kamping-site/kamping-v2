#pragma once

#include <ranges>

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <
    send_buffer                         SBuf,
    recv_buffer_v                       RBuf,
    convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void allgatherv(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Allgatherv(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        ptr(rbuf),
        std::ranges::data(counts(rbuf)),
        std::ranges::data(displs(rbuf)),
        type(rbuf),
        handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
