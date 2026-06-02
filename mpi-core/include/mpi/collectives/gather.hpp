// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <climits>

#include <mpi.h>

#include "kamping/kassert/kassert.hpp"
#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

#if MPI_VERSION >= 4
template <send_buffer SBuf, recv_buffer RBuf, rank Root, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void gather_c(SBuf&& sbuf, RBuf&& rbuf, Root root, Comm const& comm) {
    int comm_size = 0;
    int comm_rank = 0;
    MPI_Comm_size(handle(comm), &comm_size);
    MPI_Comm_rank(handle(comm), &comm_rank);
    KAMPING_ASSERT(
        to_rank(root) != comm_rank || count(rbuf) % comm_size == 0,
        "on root: recv buffer size must be divisible by comm size"
    );
    int err = MPI_Gather_c(
        ptr(sbuf),
        count(sbuf),
        type(sbuf),
        ptr(rbuf),
        count(rbuf) / comm_size,
        type(rbuf),
        to_rank(root),
        handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
#endif

template <send_buffer SBuf, recv_buffer RBuf, rank Root, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void gather(SBuf&& sbuf, RBuf&& rbuf, Root root, Comm const& comm) {
    int comm_size = 0;
    int comm_rank = 0;
    MPI_Comm_size(handle(comm), &comm_size);
    MPI_Comm_rank(handle(comm), &comm_rank);
    KAMPING_ASSERT(
        to_rank(root) != comm_rank || count(rbuf) % comm_size == 0,
        "on root: recv buffer size must be divisible by comm size"
    );
    KAMPING_ASSERT(count(sbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    KAMPING_ASSERT(count(rbuf) / comm_size <= INT_MAX, "element count exceeds int range; requires MPI-4");
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
