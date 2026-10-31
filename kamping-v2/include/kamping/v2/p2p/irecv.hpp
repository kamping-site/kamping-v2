// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/infer.hpp"
#include "kamping/v2/iresult.hpp"
#include "kamping/v2/p2p/recv.hpp"
#include "mpi/handle.hpp"
#include "mpi/p2p/imrecv.hpp"
#include "mpi/p2p/irecv.hpp"

/// @file
/// Non-blocking receives. Source and tag may be omitted (any source / any tag). Without a request, the call returns
/// an @ref kamping::v2::iresult that owns the request; with a caller-supplied request (passed first), it returns the
/// buffer and the caller must complete the request.
///
/// **Synchronous work on the caller's thread.** Before the receive is started, the `infer(comm_op::recv, ...)`
/// customization point runs. For a deferred receive buffer (e.g. `views::resize`) it performs a **blocking**
/// `MPI_Mprobe` to size the buffer, so these calls are non-blocking only for the data transfer. Pass a buffer of known
/// size to avoid this.

namespace kamping::v2 {

/// Starts receiving a message from `source` with `tag` into `rbuf`, using the caller-supplied `request` (`MPI_Irecv`,
/// or `MPI_Imrecv` for a deferred buffer). The caller must keep `rbuf` alive and complete `request`.
/// @return The receive buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request,
    mpi::experimental::recv_buffer                                RBuf,
    mpi::experimental::rank                                       Source,
    mpi::experimental::tag                                        Tag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm>
auto irecv(Request&& request, RBuf&& rbuf, Source source, Tag tag, Comm const& comm) -> RBuf {
    detail::infer_and_recv(
        rbuf,
        mpi::experimental::to_rank(source),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        [&](int src, int tg) { mpi::experimental::irecv(rbuf, src, tg, comm, std::forward<Request>(request)); },
        [&](MPI_Message* msg) { mpi::experimental::imrecv(rbuf, msg, std::forward<Request>(request)); }
    );
    return std::forward<RBuf>(rbuf);
}

/// Like the overload above, receiving with any tag.
template <
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request,
    mpi::experimental::recv_buffer                                RBuf,
    mpi::experimental::rank                                       Source,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm>
auto irecv(Request&& request, RBuf&& rbuf, Source source, Comm const& comm) -> RBuf {
    return kamping::v2::irecv(
        std::forward<Request>(request),
        std::forward<RBuf>(rbuf),
        std::move(source),
        MPI_ANY_TAG,
        comm
    );
}

/// Like the overload above, receiving from any source with any tag.
template <
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request,
    mpi::experimental::recv_buffer                                RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm>
auto irecv(Request&& request, RBuf&& rbuf, Comm const& comm) -> RBuf {
    return kamping::v2::irecv(
        std::forward<Request>(request),
        std::forward<RBuf>(rbuf),
        MPI_ANY_SOURCE,
        MPI_ANY_TAG,
        comm
    );
}

/// Starts receiving a message from `source` with `tag` into `rbuf` (`MPI_Irecv`, or `MPI_Imrecv` for a deferred
/// buffer).
/// @return An iresult owning the request (and the buffer, if passed as rvalue); complete it with `wait()` or `test()`.
template <
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::rank                                Source,
    mpi::experimental::tag                                 Tag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto irecv(RBuf&& rbuf, Source source, Tag tag, Comm const& comm) -> iresult<RBuf> {
    iresult<RBuf> res{std::forward<RBuf>(rbuf)};
    kamping::v2::irecv(res.mpi_handle_ptr(), res.view(), std::move(source), std::move(tag), comm);
    return res;
}

/// Like the overload above, receiving with any tag.
template <
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::rank                                Source,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto irecv(RBuf&& rbuf, Source source, Comm const& comm) -> iresult<RBuf> {
    return kamping::v2::irecv(std::forward<RBuf>(rbuf), std::move(source), MPI_ANY_TAG, comm);
}

/// Like the overload above, receiving from any source with any tag.
template <mpi::experimental::recv_buffer RBuf, mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto irecv(RBuf&& rbuf, Comm const& comm) -> iresult<RBuf> {
    return kamping::v2::irecv(std::forward<RBuf>(rbuf), MPI_ANY_SOURCE, MPI_ANY_TAG, comm);
}

} // namespace kamping::v2
