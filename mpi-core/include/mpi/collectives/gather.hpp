// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Gathers `sbuf` from all ranks into `rbuf` on `root` (`MPI_Gather`).
///
/// On root, the per-rank receive count is `count(rbuf) / comm_size`; `rbuf` is ignored on the other ranks.
template <send_buffer SBuf, recv_buffer RBuf, rank Root, convertible_to_mpi_handle<MPI_Comm> Comm>
void gather(SBuf&& sbuf, RBuf&& rbuf, Root root, Comm const& comm) {
    int comm_size = 0;
    int comm_rank = 0;
    MPI_Comm_size(mpi::experimental::handle(comm), &comm_size);
    MPI_Comm_rank(mpi::experimental::handle(comm), &comm_rank);
    KAMPING_V2_ASSERT(
        mpi::experimental::to_rank(root) != comm_rank
            || static_cast<int>(mpi::experimental::count(rbuf)) % comm_size == 0,
        "on root: recv buffer size must be divisible by comm size"
    );
    int err = MPI_Gather(
        mpi::experimental::ptr(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::ptr(rbuf),
        static_cast<int>(mpi::experimental::count(rbuf)) / comm_size,
        mpi::experimental::type(rbuf),
        mpi::experimental::to_rank(root),
        mpi::experimental::handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
