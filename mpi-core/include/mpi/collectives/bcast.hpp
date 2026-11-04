// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Broadcasts `send_recv_buf` from `root` to all ranks (`MPI_Bcast`). The buffer must have the same count on all ranks.
template <send_recv_buffer SRBuf, rank Root, convertible_to_mpi_handle<MPI_Comm> Comm>
void bcast(SRBuf&& send_recv_buf, Root root, Comm const& comm) {
    int err = MPI_Bcast(
        mpi::experimental::ptr(send_recv_buf),
        static_cast<int>(mpi::experimental::count(send_recv_buf)),
        mpi::experimental::type(send_recv_buf),
        mpi::experimental::to_rank(root),
        mpi::experimental::handle(comm)

    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
