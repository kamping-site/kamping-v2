// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Starts sending `sbuf` to `dest` (`MPI_Isend`). The caller must keep `sbuf` alive and complete `request`.
template <
    send_buffer                                SBuf,
    rank                                       Dest,
    tag                                        Tag,
    convertible_to_mpi_handle<MPI_Comm>        Comm,
    convertible_to_mpi_handle_ptr<MPI_Request> Request>
void isend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Isend(
        mpi::experimental::ptr(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        mpi::experimental::handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// Starts sending `sbuf` to `dest` in buffered mode (`MPI_Ibsend`). The caller must keep `sbuf` alive and complete
/// `request`.
template <
    send_buffer                                SBuf,
    rank                                       Dest,
    tag                                        Tag,
    convertible_to_mpi_handle<MPI_Comm>        Comm,
    convertible_to_mpi_handle_ptr<MPI_Request> Request>
void ibsend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Ibsend(
        mpi::experimental::ptr(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        mpi::experimental::handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// Starts sending `sbuf` to `dest` in synchronous mode (`MPI_Issend`). The caller must keep `sbuf` alive and complete
/// `request`.
template <
    send_buffer                                SBuf,
    rank                                       Dest,
    tag                                        Tag,
    convertible_to_mpi_handle<MPI_Comm>        Comm,
    convertible_to_mpi_handle_ptr<MPI_Request> Request>
void issend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Issend(
        mpi::experimental::ptr(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        mpi::experimental::handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// Starts sending `sbuf` to `dest` in ready mode (`MPI_Irsend`). The caller must keep `sbuf` alive and complete
/// `request`.
template <
    send_buffer                                SBuf,
    rank                                       Dest,
    tag                                        Tag,
    convertible_to_mpi_handle<MPI_Comm>        Comm,
    convertible_to_mpi_handle_ptr<MPI_Request> Request>
void irsend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Irsend(
        mpi::experimental::ptr(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        mpi::experimental::handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
