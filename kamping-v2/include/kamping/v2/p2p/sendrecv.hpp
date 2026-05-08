// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "kamping/v2/infer.hpp"
#include "kamping/v2/p2p/constants.hpp"
#include "kamping/v2/result.hpp"
#include "mpi/handle.hpp"
#include "mpi/p2p/sendrecv.hpp"

namespace kamping::v2 {

template <
    mpi::experimental::send_buffer                               SBuf,
    mpi::experimental::recv_buffer                               RBuf,
    mpi::experimental::rank                                      Dest    = int,
    mpi::experimental::rank                                      Source  = int,
    mpi::experimental::tag                                       SendTag = int,
    mpi::experimental::tag                                       RecvTag = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm    = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status  = MPI_Status*>
auto sendrecv(
    SBuf&&      sbuf,
    Dest        dest,
    SendTag     send_tag,
    RBuf&&      rbuf,
    Source      source   = MPI_ANY_SOURCE,
    RecvTag     recv_tag = MPI_ANY_TAG,
    Comm const& comm     = MPI_COMM_WORLD,
    Status&&    status   = MPI_STATUS_IGNORE
) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(
        comm_op::sendrecv{},
        res.send,
        res.recv,
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(send_tag),
        mpi::experimental::to_rank(source),
        mpi::experimental::to_tag(recv_tag),
        mpi::experimental::handle(comm)
    );
    mpi::experimental::sendrecv(
        res.send,
        std::move(dest),
        std::move(send_tag),
        res.recv,
        std::move(source),
        std::move(recv_tag),
        comm,
        std::forward<Status>(status)
    );
    return res;
}

template <
    mpi::experimental::send_buffer                               SBuf,
    mpi::experimental::recv_buffer                               RBuf,
    mpi::experimental::rank                                      Dest   = int,
    mpi::experimental::rank                                      Source = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm   = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto sendrecv(
    SBuf&&      sbuf,
    Dest        dest,
    RBuf&&      rbuf,
    Source      source = MPI_ANY_SOURCE,
    Comm const& comm   = MPI_COMM_WORLD,
    Status&&    status = MPI_STATUS_IGNORE
) -> result<SBuf, RBuf> {
    return kamping::v2::sendrecv(
        std::forward<SBuf>(sbuf),
        std::move(dest),
        DEFAULT_SEND_TAG,
        std::forward<RBuf>(rbuf),
        std::move(source),
        DEFAULT_SEND_TAG,
        comm,
        std::forward<Status>(status)
    );
}

} // namespace kamping::v2
