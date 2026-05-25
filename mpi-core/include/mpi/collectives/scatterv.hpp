// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <climits>
#include <ranges>

#include <mpi.h>

#include "kamping/kassert/kassert.hpp"
#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <send_buffer_v SBuf, recv_buffer RBuf, rank Root, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void scatterv(SBuf&& sbuf, RBuf&& rbuf, Root root, Comm const& comm) {
    KAMPING_ASSERT(count(rbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    int err = MPI_Scatterv(
        ptr(sbuf),
        std::ranges::data(counts(sbuf)),
        std::ranges::data(displs(sbuf)),
        type(sbuf),
        ptr(rbuf),
        static_cast<int>(count(rbuf)),
        type(rbuf),
        to_rank(root),
        handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
