// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/infer.hpp"
#include "kamping/v2/p2p/constants.hpp"
#include "kamping/v2/result.hpp"
#include "mpi/handle.hpp"
#include "mpi/p2p/sendrecv.hpp"

namespace kamping::v2 {

/// Sends `sbuf` to `dest` with `send_tag` and receives into `rbuf` from `source` with `recv_tag` in one call
/// (`MPI_Sendrecv`).
///
/// If `rbuf` is a deferred receive buffer (e.g. `views::resize`), the counts are exchanged first and `rbuf` is resized.
/// @param status Optional: filled with the status of the received message. Defaults to `MPI_STATUS_IGNORE`.
/// @return The send and receive buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_buffer                               SBuf,
    mpi::experimental::recv_buffer                               RBuf,
    mpi::experimental::rank                                      Dest,
    mpi::experimental::rank                                      Source,
    mpi::experimental::tag                                       SendTag,
    mpi::experimental::tag                                       RecvTag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto sendrecv(
    SBuf&&      sbuf,
    Dest        dest,
    SendTag     send_tag,
    RBuf&&      rbuf,
    Source      source,
    RecvTag     recv_tag,
    Comm const& comm,
    Status&&    status = MPI_STATUS_IGNORE
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

/// Like the overload above, sending with @ref DEFAULT_SEND_TAG and receiving with any tag.
template <
    mpi::experimental::send_buffer                               SBuf,
    mpi::experimental::recv_buffer                               RBuf,
    mpi::experimental::rank                                      Dest,
    mpi::experimental::rank                                      Source,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto sendrecv(SBuf&& sbuf, Dest dest, RBuf&& rbuf, Source source, Comm const& comm, Status&& status = MPI_STATUS_IGNORE)
    -> result<SBuf, RBuf> {
    return kamping::v2::sendrecv(
        std::forward<SBuf>(sbuf),
        std::move(dest),
        DEFAULT_SEND_TAG,
        std::forward<RBuf>(rbuf),
        std::move(source),
        MPI_ANY_TAG,
        comm,
        std::forward<Status>(status)
    );
}

/// Like the overload above, receiving from any source.
template <
    mpi::experimental::send_buffer                               SBuf,
    mpi::experimental::recv_buffer                               RBuf,
    mpi::experimental::rank                                      Dest,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto sendrecv(SBuf&& sbuf, Dest dest, RBuf&& rbuf, Comm const& comm, Status&& status = MPI_STATUS_IGNORE)
    -> result<SBuf, RBuf> {
    return kamping::v2::sendrecv(
        std::forward<SBuf>(sbuf),
        std::move(dest),
        DEFAULT_SEND_TAG,
        std::forward<RBuf>(rbuf),
        MPI_ANY_SOURCE,
        MPI_ANY_TAG,
        comm,
        std::forward<Status>(status)
    );
}

} // namespace kamping::v2
