// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Sends `sbuf` to `dest` (`MPI_Send`).
template <send_buffer SBuf, rank Dest, tag Tag, convertible_to_mpi_handle<MPI_Comm> Comm>
void send(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm) {
    int err = MPI_Send(
        mpi::experimental::ptr(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// Sends `sbuf` to `dest` in buffered mode (`MPI_Bsend`).
template <send_buffer SBuf, rank Dest, tag Tag, convertible_to_mpi_handle<MPI_Comm> Comm>
void bsend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm) {
    int err = MPI_Bsend(
        mpi::experimental::ptr(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// Sends `sbuf` to `dest` in synchronous mode (`MPI_Ssend`).
template <send_buffer SBuf, rank Dest, tag Tag, convertible_to_mpi_handle<MPI_Comm> Comm>
void ssend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm) {
    int err = MPI_Ssend(
        mpi::experimental::ptr(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// Sends `sbuf` to `dest` in ready mode (`MPI_Rsend`).
template <send_buffer SBuf, rank Dest, tag Tag, convertible_to_mpi_handle<MPI_Comm> Comm>
void rsend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm) {
    int err = MPI_Rsend(
        mpi::experimental::ptr(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
