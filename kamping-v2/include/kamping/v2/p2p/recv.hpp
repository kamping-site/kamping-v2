// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <type_traits>
#include <utility>

#include <mpi.h>

#include "kamping/v2/infer.hpp"
#include "mpi/handle.hpp"
#include "mpi/p2p/mrecv.hpp"
#include "mpi/p2p/recv.hpp"

namespace kamping::v2 {

namespace detail {
/// Runs infer() for a receive into `rbuf`, then issues the receive via `recv_fn(src, tag)` or, if infer() matched the
/// message, via `mrecv_fn(message)`.
///
/// infer() returns the matched `MPI_Message` for a deferred buffer (`MPI_Mprobe`), the resolved `{source, tag}` if
/// the buffer does not support matched probes (`MPI_Probe`, so the receive must target exactly that message), or
/// nothing for a buffer of known size.
template <typename RBuf, typename RecvFn, typename MrecvFn>
void infer_and_recv(RBuf& rbuf, int source, int tag, MPI_Comm comm, RecvFn&& recv_fn, MrecvFn&& mrecv_fn) {
    using infer_result_t = decltype(infer(comm_op::recv{}, rbuf, source, tag, comm));
    if constexpr (std::is_same_v<infer_result_t, MPI_Message>) {
        MPI_Message msg = infer(comm_op::recv{}, rbuf, source, tag, comm);
        mrecv_fn(&msg);
    } else if constexpr (std::is_same_v<infer_result_t, std::pair<int, int>>) {
        auto [resolved_source, resolved_tag] = infer(comm_op::recv{}, rbuf, source, tag, comm);
        recv_fn(resolved_source, resolved_tag);
    } else {
        infer(comm_op::recv{}, rbuf, source, tag, comm);
        recv_fn(source, tag);
    }
}
} // namespace detail

/// Receives a message from `source` with `tag` into `rbuf` (`MPI_Recv`).
///
/// If `rbuf` is a deferred receive buffer (e.g. `views::resize` or `auto_recv<T>()`), the message is matched with
/// `MPI_Mprobe` first and `rbuf` is resized to fit, then received with `MPI_Mrecv`.
/// @param status Optional: filled with the status of the received message. Defaults to `MPI_STATUS_IGNORE`.
/// @return The receive buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::recv_buffer                               RBuf,
    mpi::experimental::rank                                      Source,
    mpi::experimental::tag                                       Tag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto recv(RBuf&& rbuf, Source source, Tag tag, Comm const& comm, Status&& status = MPI_STATUS_IGNORE) -> RBuf {
    detail::infer_and_recv(
        rbuf,
        mpi::experimental::to_rank(source),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        [&](int src, int tg) { mpi::experimental::recv(rbuf, src, tg, comm, std::forward<Status>(status)); },
        [&](MPI_Message* msg) { mpi::experimental::mrecv(rbuf, msg, std::forward<Status>(status)); }
    );
    return std::forward<RBuf>(rbuf);
}

/// Receives a message from `source` with any tag into `rbuf`; see the overload above.
template <
    mpi::experimental::recv_buffer                               RBuf,
    mpi::experimental::rank                                      Source,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto recv(RBuf&& rbuf, Source source, Comm const& comm, Status&& status = MPI_STATUS_IGNORE) -> RBuf {
    return kamping::v2::recv(
        std::forward<RBuf>(rbuf),
        std::move(source),
        MPI_ANY_TAG,
        comm,
        std::forward<Status>(status)
    );
}

/// Receives a message from any source with any tag into `rbuf`; see the overload above.
template <
    mpi::experimental::recv_buffer                               RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto recv(RBuf&& rbuf, Comm const& comm, Status&& status = MPI_STATUS_IGNORE) -> RBuf {
    return kamping::v2::recv(std::forward<RBuf>(rbuf), MPI_ANY_SOURCE, MPI_ANY_TAG, comm, std::forward<Status>(status));
}

} // namespace kamping::v2
