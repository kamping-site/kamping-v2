// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <ranges>

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Exchanges variable amounts of data between all ranks, as described by the counts and displacements of the variadic
/// buffers `sbuf` and `rbuf` (`MPI_Alltoallv`).
template <send_buffer_v SBuf, recv_buffer_v RBuf, convertible_to_mpi_handle<MPI_Comm> Comm>
void alltoallv(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm) {
    int err = MPI_Alltoallv(
        mpi::experimental::ptr(sbuf),
        std::ranges::data(mpi::experimental::counts(sbuf)),
        std::ranges::data(mpi::experimental::displs(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::ptr(rbuf),
        std::ranges::data(mpi::experimental::counts(rbuf)),
        std::ranges::data(mpi::experimental::displs(rbuf)),
        mpi::experimental::type(rbuf),
        mpi::experimental::handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
