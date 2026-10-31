// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#if MPI_VERSION >= 4

    #include <utility>

    #include "kamping/v2/infer.hpp"
    #include "kamping/v2/iresult.hpp"
    #include "kamping/v2/p2p/constants.hpp"
    #include "kamping/v2/result.hpp"
    #include "mpi/handle.hpp"
    #include "mpi/p2p/isendrecv.hpp"

/// @file
/// Non-blocking combined send and receive (MPI 4.0). The forms mirror @ref kamping::v2::sendrecv. Without a request,
/// the call returns an @ref kamping::v2::iresult that owns the request; with a caller-supplied request (passed first),
/// it returns the buffers and the caller must complete the request.
///
/// **Synchronous work on the caller's thread.** Before `MPI_Isendrecv` is issued, the
/// `infer(comm_op::sendrecv, ...)` customization point runs. For a deferred receive buffer (e.g. `views::resize`) it
/// performs a **blocking** `MPI_Sendrecv` of the counts to size the buffer. Pass a buffer of known size to avoid this.

namespace kamping::v2 {

/// Starts sending `sbuf` to `dest` with `send_tag` and receiving into `rbuf` from `source` with `recv_tag`, using the
/// caller-supplied `request` (`MPI_Isendrecv`). The caller must keep both buffers alive and complete `request`.
/// @return The send and receive buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request,
    mpi::experimental::send_buffer                                SBuf,
    mpi::experimental::recv_buffer                                RBuf,
    mpi::experimental::rank                                       Dest,
    mpi::experimental::rank                                       Source,
    mpi::experimental::tag                                        SendTag,
    mpi::experimental::tag                                        RecvTag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm>
auto isendrecv(
    Request&&   request,
    SBuf&&      sbuf,
    Dest        dest,
    SendTag     send_tag,
    RBuf&&      rbuf,
    Source      source,
    RecvTag     recv_tag,
    Comm const& comm
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
    mpi::experimental::isendrecv(
        res.send,
        std::move(dest),
        std::move(send_tag),
        res.recv,
        std::move(source),
        std::move(recv_tag),
        comm,
        std::forward<Request>(request)
    );
    return res;
}

/// Like the overload above, sending with @ref DEFAULT_SEND_TAG and receiving with any tag.
template <
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request,
    mpi::experimental::send_buffer                                SBuf,
    mpi::experimental::recv_buffer                                RBuf,
    mpi::experimental::rank                                       Dest,
    mpi::experimental::rank                                       Source,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm>
auto isendrecv(Request&& request, SBuf&& sbuf, Dest dest, RBuf&& rbuf, Source source, Comm const& comm)
    -> result<SBuf, RBuf> {
    return kamping::v2::isendrecv(
        std::forward<Request>(request),
        std::forward<SBuf>(sbuf),
        std::move(dest),
        DEFAULT_SEND_TAG,
        std::forward<RBuf>(rbuf),
        std::move(source),
        MPI_ANY_TAG,
        comm
    );
}

/// Like the overload above, receiving from any source.
template <
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request,
    mpi::experimental::send_buffer                                SBuf,
    mpi::experimental::recv_buffer                                RBuf,
    mpi::experimental::rank                                       Dest,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm>
auto isendrecv(Request&& request, SBuf&& sbuf, Dest dest, RBuf&& rbuf, Comm const& comm) -> result<SBuf, RBuf> {
    return kamping::v2::isendrecv(
        std::forward<Request>(request),
        std::forward<SBuf>(sbuf),
        std::move(dest),
        DEFAULT_SEND_TAG,
        std::forward<RBuf>(rbuf),
        MPI_ANY_SOURCE,
        MPI_ANY_TAG,
        comm
    );
}

/// Starts sending `sbuf` to `dest` with `send_tag` and receiving into `rbuf` from `source` with `recv_tag`
/// (`MPI_Isendrecv`).
/// @return An iresult owning the request (and the buffers passed as rvalue); complete it with `wait()` or `test()`.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::rank                                Dest,
    mpi::experimental::rank                                Source,
    mpi::experimental::tag                                 SendTag,
    mpi::experimental::tag                                 RecvTag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto isendrecv(SBuf&& sbuf, Dest dest, SendTag send_tag, RBuf&& rbuf, Source source, RecvTag recv_tag, Comm const& comm)
    -> iresult<SBuf, RBuf> {
    iresult<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    kamping::v2::isendrecv(
        res.mpi_handle_ptr(),
        res.send(),
        std::move(dest),
        std::move(send_tag),
        res.recv(),
        std::move(source),
        std::move(recv_tag),
        comm
    );
    return res;
}

/// Like the overload above, sending with @ref DEFAULT_SEND_TAG and receiving with any tag.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::rank                                Dest,
    mpi::experimental::rank                                Source,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto isendrecv(SBuf&& sbuf, Dest dest, RBuf&& rbuf, Source source, Comm const& comm) -> iresult<SBuf, RBuf> {
    return kamping::v2::isendrecv(
        std::forward<SBuf>(sbuf),
        std::move(dest),
        DEFAULT_SEND_TAG,
        std::forward<RBuf>(rbuf),
        std::move(source),
        MPI_ANY_TAG,
        comm
    );
}

/// Like the overload above, receiving from any source.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::rank                                Dest,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto isendrecv(SBuf&& sbuf, Dest dest, RBuf&& rbuf, Comm const& comm) -> iresult<SBuf, RBuf> {
    return kamping::v2::isendrecv(
        std::forward<SBuf>(sbuf),
        std::move(dest),
        DEFAULT_SEND_TAG,
        std::forward<RBuf>(rbuf),
        MPI_ANY_SOURCE,
        MPI_ANY_TAG,
        comm
    );
}

} // namespace kamping::v2

#endif // MPI_VERSION >= 4
