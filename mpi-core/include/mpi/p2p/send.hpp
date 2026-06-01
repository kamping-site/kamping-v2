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
template <send_buffer SBuf, rank Dest = int, tag Tag = int, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void send_c(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Send_c(ptr(sbuf), count(sbuf), type(sbuf), to_rank(dest), to_tag(tag), handle(comm));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <send_buffer SBuf, rank Dest = int, tag Tag = int, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void bsend_c(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Bsend_c(ptr(sbuf), count(sbuf), type(sbuf), to_rank(dest), to_tag(tag), handle(comm));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <send_buffer SBuf, rank Dest = int, tag Tag = int, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void ssend_c(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Ssend_c(ptr(sbuf), count(sbuf), type(sbuf), to_rank(dest), to_tag(tag), handle(comm));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <send_buffer SBuf, rank Dest = int, tag Tag = int, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void rsend_c(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Rsend_c(ptr(sbuf), count(sbuf), type(sbuf), to_rank(dest), to_tag(tag), handle(comm));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
#endif

template <send_buffer SBuf, rank Dest = int, tag Tag = int, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void send(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    KAMPING_ASSERT(count(sbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    int err = MPI_Send(ptr(sbuf), static_cast<int>(count(sbuf)), type(sbuf), to_rank(dest), to_tag(tag), handle(comm));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <send_buffer SBuf, rank Dest = int, tag Tag = int, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void bsend(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    KAMPING_ASSERT(count(sbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    int err = MPI_Bsend(ptr(sbuf), static_cast<int>(count(sbuf)), type(sbuf), to_rank(dest), to_tag(tag), handle(comm));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <send_buffer SBuf, rank Dest = int, tag Tag = int, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void ssend(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    KAMPING_ASSERT(count(sbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    int err = MPI_Ssend(ptr(sbuf), static_cast<int>(count(sbuf)), type(sbuf), to_rank(dest), to_tag(tag), handle(comm));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <send_buffer SBuf, rank Dest = int, tag Tag = int, convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void rsend(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    KAMPING_ASSERT(count(sbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    int err = MPI_Rsend(ptr(sbuf), static_cast<int>(count(sbuf)), type(sbuf), to_rank(dest), to_tag(tag), handle(comm));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

} // namespace mpi::experimental
