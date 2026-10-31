// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <ranges>

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Distributes parts of the variadic buffer `sbuf` on `root` to all ranks, receiving into `rbuf` (`MPI_Scatterv`).
/// `sbuf` is ignored on the other ranks.
template <send_buffer_v SBuf, recv_buffer RBuf, rank Root, convertible_to_mpi_handle<MPI_Comm> Comm>
void scatterv(SBuf&& sbuf, RBuf&& rbuf, Root root, Comm const& comm) {
    int err = MPI_Scatterv(
        mpi::experimental::ptr(sbuf),
        std::ranges::data(mpi::experimental::counts(sbuf)),
        std::ranges::data(mpi::experimental::displs(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::ptr(rbuf),
        static_cast<int>(mpi::experimental::count(rbuf)),
        mpi::experimental::type(rbuf),
        mpi::experimental::to_rank(root),
        mpi::experimental::handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
