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
template <send_buffer SBuf, recv_buffer RBuf, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void allgather(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) {
    int comm_size = 0;
    MPI_Comm_size(handle(comm), &comm_size);
    KAMPING_ASSERT(count(rbuf) % comm_size == 0, "recv buffer size must be divisible by comm size");
#if MPI_VERSION >= 4
    int err = MPI_Allgather_c(
        ptr(sbuf),
        count(sbuf),
        type(sbuf),
        ptr(rbuf),
        count(rbuf) / comm_size,
        type(rbuf),
        handle(comm)
    );
#else
    KAMPING_ASSERT(count(sbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    KAMPING_ASSERT(count(rbuf) / comm_size <= INT_MAX, "element count exceeds int range; requires MPI-4");
    int err = MPI_Allgather(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        ptr(rbuf),
        static_cast<int>(count(rbuf)) / comm_size,
        type(rbuf),
        handle(comm)
    );
#endif
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
