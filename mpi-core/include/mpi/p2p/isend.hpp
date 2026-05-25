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
template <
    send_buffer                                SBuf,
    rank                                       Dest    = int,
    tag                                        Tag     = int,
    convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void isend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
#if MPI_VERSION >= 4
    int err =
        MPI_Isend_c(ptr(sbuf), count(sbuf), type(sbuf), to_rank(dest), to_tag(tag), handle(comm), handle_ptr(request));
#else
    KAMPING_ASSERT(count(sbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    int err = MPI_Isend(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        to_rank(dest),
        to_tag(tag),
        handle(comm),
        handle_ptr(request)
    );
#endif
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    send_buffer                                SBuf,
    rank                                       Dest    = int,
    tag                                        Tag     = int,
    convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void ibsend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
#if MPI_VERSION >= 4
    int err =
        MPI_Ibsend_c(ptr(sbuf), count(sbuf), type(sbuf), to_rank(dest), to_tag(tag), handle(comm), handle_ptr(request));
#else
    KAMPING_ASSERT(count(sbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    int err = MPI_Ibsend(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        to_rank(dest),
        to_tag(tag),
        handle(comm),
        handle_ptr(request)
    );
#endif
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    send_buffer                                SBuf,
    rank                                       Dest    = int,
    tag                                        Tag     = int,
    convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void issend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
#if MPI_VERSION >= 4
    int err =
        MPI_Issend_c(ptr(sbuf), count(sbuf), type(sbuf), to_rank(dest), to_tag(tag), handle(comm), handle_ptr(request));
#else
    KAMPING_ASSERT(count(sbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    int err = MPI_Issend(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        to_rank(dest),
        to_tag(tag),
        handle(comm),
        handle_ptr(request)
    );
#endif
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    send_buffer                                SBuf,
    rank                                       Dest    = int,
    tag                                        Tag     = int,
    convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void irsend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
#if MPI_VERSION >= 4
    int err =
        MPI_Irsend_c(ptr(sbuf), count(sbuf), type(sbuf), to_rank(dest), to_tag(tag), handle(comm), handle_ptr(request));
#else
    KAMPING_ASSERT(count(sbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    int err = MPI_Irsend(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        to_rank(dest),
        to_tag(tag),
        handle(comm),
        handle_ptr(request)
    );
#endif
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
