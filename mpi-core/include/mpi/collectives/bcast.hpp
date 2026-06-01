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
template <send_recv_buffer SRBuf, rank Root = int, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void bcast_c(SRBuf&& send_recv_buf, Root root = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Bcast_c(ptr(send_recv_buf), count(send_recv_buf), type(send_recv_buf), to_rank(root), handle(comm));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
#endif

template <send_recv_buffer SRBuf, rank Root = int, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void bcast(SRBuf&& send_recv_buf, Root root = 0, Comm const& comm = MPI_COMM_WORLD) {
    KAMPING_ASSERT(count(send_recv_buf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
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
