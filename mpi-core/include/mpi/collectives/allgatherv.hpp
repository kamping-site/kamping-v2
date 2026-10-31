// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <ranges>

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Gathers `sbuf` from all ranks into the variadic buffer `rbuf` on every rank (`MPI_Allgatherv`).
template <send_buffer SBuf, recv_buffer_v RBuf, convertible_to_mpi_handle<MPI_Comm> Comm>
void allgatherv(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm) {
    int err = MPI_Allgatherv(
        mpi::experimental::ptr(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
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
