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
    send_buffer                                SBuf,
    recv_buffer                                RBuf,
    rank                                       Dest    = int,
    rank                                       Source  = int,
    tag                                        SendTag = int,
    tag                                        RecvTag = int,
    convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void isendrecv_c(
    SBuf&&      sbuf,
    Dest        dest,
    SendTag     send_tag,
    RBuf&&      rbuf,
    Source      source,
    RecvTag     recv_tag,
    Comm const& comm,
    Request&&   request
) {
    int err = MPI_Isendrecv_c(
        ptr(sbuf),
        count(sbuf),
        type(sbuf),
        to_rank(dest),
        to_tag(send_tag),
        ptr(rbuf),
        count(rbuf),
        type(rbuf),
        to_rank(source),
        to_tag(recv_tag),
        handle(comm),
        handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
#endif

template <
    send_buffer                                SBuf,
    recv_buffer                                RBuf,
    rank                                       Dest    = int,
    rank                                       Source  = int,
    tag                                        SendTag = int,
    tag                                        RecvTag = int,
    convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void isendrecv(
    SBuf&&      sbuf,
    Dest        dest,
    SendTag     send_tag,
    RBuf&&      rbuf,
    Source      source,
    RecvTag     recv_tag,
    Comm const& comm,
    Request&&   request
) {
    KAMPING_ASSERT(count(sbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    KAMPING_ASSERT(count(rbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    int err = MPI_Isendrecv(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        to_rank(dest),
        to_tag(send_tag),
        ptr(rbuf),
        static_cast<int>(count(rbuf)),
        type(rbuf),
        to_rank(source),
        to_tag(recv_tag),
        handle(comm),
        handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

} // namespace mpi::experimental
