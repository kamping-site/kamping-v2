// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Sends an equal share of `sbuf` to every rank and receives one share from every rank into `rbuf` (`MPI_Alltoall`).
///
/// The per-rank counts are `count(sbuf) / comm_size` and `count(rbuf) / comm_size`.
template <send_buffer SBuf, recv_buffer RBuf, convertible_to_mpi_handle<MPI_Comm> Comm>
void alltoall(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm) {
    int comm_size = 0;
    MPI_Comm_size(mpi::experimental::handle(comm), &comm_size);
    using scount_t = decltype(mpi::experimental::count(sbuf));
    using rcount_t = decltype(mpi::experimental::count(rbuf));
    KAMPING_V2_ASSERT(
        mpi::experimental::count(sbuf) % static_cast<scount_t>(comm_size) == scount_t{0},
        "send buffer size must be divisible by comm size"
    );
    KAMPING_V2_ASSERT(
        mpi::experimental::count(rbuf) % static_cast<rcount_t>(comm_size) == rcount_t{0},
        "recv buffer size must be divisible by comm size"
    );
    int err = MPI_Alltoall(
        mpi::experimental::ptr(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)) / comm_size,
        mpi::experimental::type(sbuf),
        mpi::experimental::ptr(rbuf),
        static_cast<int>(mpi::experimental::count(rbuf)) / comm_size,
        mpi::experimental::type(rbuf),
        mpi::experimental::handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
