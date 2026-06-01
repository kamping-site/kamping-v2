// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <ranges>

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

#if MPI_VERSION >= 4
template <send_buffer_v_c SBuf, recv_buffer_v_c RBuf, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void alltoallv_c(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Alltoallv_c(
        ptr(sbuf),
        std::ranges::data(counts(sbuf)),
        std::ranges::data(displs(sbuf)),
        type(sbuf),
        ptr(rbuf),
        std::ranges::data(counts(rbuf)),
        std::ranges::data(displs(rbuf)),
        type(rbuf),
        handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
#endif

template <send_buffer_v SBuf, recv_buffer_v RBuf, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void alltoallv(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Alltoallv(
        ptr(sbuf),
        std::ranges::data(counts(sbuf)),
        std::ranges::data(displs(sbuf)),
        type(sbuf),
        ptr(rbuf),
        std::ranges::data(counts(rbuf)),
        std::ranges::data(displs(rbuf)),
        type(rbuf),
        handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
