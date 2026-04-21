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
    rank                                Root,
    convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void gatherv(SBuf&& sbuf, RBuf&& rbuf, Root root, Comm const& comm) {
    int err = MPI_Gatherv(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        ptr(rbuf),
        std::ranges::data(counts(rbuf)),
        std::ranges::data(displs(rbuf)),
        type(rbuf),
        to_rank(root),
        handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
