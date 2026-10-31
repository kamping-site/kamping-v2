// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Gathers `sbuf` from all ranks into `rbuf` on every rank (`MPI_Allgather`).
///
/// The per-rank receive count is `count(rbuf) / comm_size`, so `rbuf` must already have room for all ranks.
template <send_buffer SBuf, recv_buffer RBuf, convertible_to_mpi_handle<MPI_Comm> Comm>
void allgather(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm) {
    int comm_size = 0;
    MPI_Comm_size(mpi::experimental::handle(comm), &comm_size);
    KAMPING_V2_ASSERT(
        static_cast<int>(mpi::experimental::count(rbuf)) % comm_size == 0,
        "recv buffer size must be divisible by comm size"
    );
    int err = MPI_Allgather(
        mpi::experimental::ptr(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
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
