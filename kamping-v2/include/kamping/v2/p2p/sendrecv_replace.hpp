// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/infer.hpp"
#include "kamping/v2/p2p/constants.hpp"
#include "mpi/handle.hpp"
#include "mpi/p2p/sendrecv_replace.hpp"

namespace kamping::v2 {

/// Sends the contents of `send_recv_buf` to `dest` with `send_tag` and replaces them with a message received from
/// `source` with `recv_tag`, in one call (`MPI_Sendrecv_replace`).
///
/// The same buffer, count and datatype are used for both directions: `count(send_recv_buf)` elements are sent, and the
/// received message must not be longer. The default `infer()` therefore rejects deferred receive buffers such as
/// `views::resize`; use `sendrecv` with a separate receive buffer for messages of unknown size.
/// @param status Optional: filled with the status of the received message. Defaults to `MPI_STATUS_IGNORE`.
/// @return The buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_recv_buffer                          SRBuf,
    mpi::experimental::rank                                      Dest,
    mpi::experimental::rank                                      Source,
    mpi::experimental::tag                                       SendTag,
    mpi::experimental::tag                                       RecvTag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto sendrecv_replace(
    SRBuf&&     send_recv_buf,
    Dest        dest,
    SendTag     send_tag,
    Source      source,
    RecvTag     recv_tag,
    Comm const& comm,
    Status&&    status = MPI_STATUS_IGNORE
) -> SRBuf {
    infer(
        comm_op::sendrecv_replace{},
        send_recv_buf,
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(send_tag),
        mpi::experimental::to_rank(source),
        mpi::experimental::to_tag(recv_tag),
        mpi::experimental::handle(comm)
    );
    mpi::experimental::sendrecv_replace(
        send_recv_buf,
        std::move(dest),
        std::move(send_tag),
        std::move(source),
        std::move(recv_tag),
        comm,
        std::forward<Status>(status)
    );
    return std::forward<SRBuf>(send_recv_buf);
}

/// Like the overload above, sending with @ref DEFAULT_SEND_TAG and receiving with any tag.
template <
    mpi::experimental::send_recv_buffer                          SRBuf,
    mpi::experimental::rank                                      Dest,
    mpi::experimental::rank                                      Source,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto sendrecv_replace(
    SRBuf&& send_recv_buf, Dest dest, Source source, Comm const& comm, Status&& status = MPI_STATUS_IGNORE
) -> SRBuf {
    return kamping::v2::sendrecv_replace(
        std::forward<SRBuf>(send_recv_buf),
        std::move(dest),
        DEFAULT_SEND_TAG,
        std::move(source),
        MPI_ANY_TAG,
        comm,
        std::forward<Status>(status)
    );
}

/// Like the overload above, receiving from any source.
template <
    mpi::experimental::send_recv_buffer                          SRBuf,
    mpi::experimental::rank                                      Dest,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto sendrecv_replace(SRBuf&& send_recv_buf, Dest dest, Comm const& comm, Status&& status = MPI_STATUS_IGNORE)
    -> SRBuf {
    return kamping::v2::sendrecv_replace(
        std::forward<SRBuf>(send_recv_buf),
        std::move(dest),
        DEFAULT_SEND_TAG,
        MPI_ANY_SOURCE,
        MPI_ANY_TAG,
        comm,
        std::forward<Status>(status)
    );
}

} // namespace kamping::v2
