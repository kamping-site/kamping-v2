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
template <
    recv_buffer                               RBuf,
    rank                                      Source = int,
    tag                                       Tag    = int,
    convertible_to_mpi_handle<MPI_Comm>       Comm   = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
void recv_c(
    RBuf&&      rbuf,
    Source      source = MPI_ANY_SOURCE,
    Tag         tag    = MPI_ANY_TAG,
    Comm const& comm   = MPI_COMM_WORLD,
    Status&&    status = MPI_STATUS_IGNORE
) {
    int err =
        MPI_Recv_c(ptr(rbuf), count(rbuf), type(rbuf), to_rank(source), to_tag(tag), handle(comm), handle_ptr(status));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
#endif

template <
    recv_buffer                               RBuf,
    rank                                      Source = int,
    tag                                       Tag    = int,
    convertible_to_mpi_handle<MPI_Comm>       Comm   = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
void recv(
    RBuf&&      rbuf,
    Source      source = MPI_ANY_SOURCE,
    Tag         tag    = MPI_ANY_TAG,
    Comm const& comm   = MPI_COMM_WORLD,
    Status&&    status = MPI_STATUS_IGNORE
) {
    KAMPING_ASSERT(count(rbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    int err = MPI_Recv(
        ptr(rbuf),
        static_cast<int>(count(rbuf)),
        type(rbuf),
        to_rank(source),
        to_tag(tag),
        handle(comm),
        handle_ptr(status)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

} // namespace mpi::experimental
