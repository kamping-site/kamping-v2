// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <type_traits>
#include <utility>

#include <mpi.h>

#include "kamping/v2/iresult.hpp"
#include "kamping/v2/p2p/constants.hpp"
#include "kamping/v2/p2p/send_mode.hpp"
#include "mpi/handle.hpp"
#include "mpi/p2p/isend.hpp"

/// @file
/// Non-blocking sends. Each form exists with and without a leading send mode, and with and without a tag (which then
/// is @ref kamping::v2::DEFAULT_SEND_TAG). Without a request, the call returns an @ref kamping::v2::iresult that owns
/// the request; with a caller-supplied request (passed first), it returns the buffer and the caller must complete the
/// request.

namespace kamping::v2 {

/// Starts sending `sbuf` to `dest` with `tag` in the given @ref send_mode, using the caller-supplied `request`
/// (`MPI_Isend`, `MPI_Ibsend`, `MPI_Issend` or `MPI_Irsend`). The caller must keep `sbuf` alive and complete `request`.
/// @return The send buffer (owned if passed as rvalue, borrowed otherwise).
template <
    is_send_mode                                                  SendMode,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request,
    mpi::experimental::send_buffer                                SBuf,
    mpi::experimental::rank                                       Dest,
    mpi::experimental::tag                                        Tag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm>
auto isend(SendMode&&, Request&& request, SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm) -> SBuf {
    using mode_t = std::remove_cvref_t<SendMode>;
    if constexpr (std::same_as<mode_t, send_mode::standard_t>) {
        mpi::experimental::isend(sbuf, std::move(dest), std::move(tag), comm, std::forward<Request>(request));
    } else if constexpr (std::same_as<mode_t, send_mode::buffered_t>) {
        mpi::experimental::ibsend(sbuf, std::move(dest), std::move(tag), comm, std::forward<Request>(request));
    } else if constexpr (std::same_as<mode_t, send_mode::sync_t>) {
        mpi::experimental::issend(sbuf, std::move(dest), std::move(tag), comm, std::forward<Request>(request));
    } else {
        mpi::experimental::irsend(sbuf, std::move(dest), std::move(tag), comm, std::forward<Request>(request));
    }
    return std::forward<SBuf>(sbuf);
}

/// Like the overload above, with @ref DEFAULT_SEND_TAG.
template <
    is_send_mode                                                  SendMode,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request,
    mpi::experimental::send_buffer                                SBuf,
    mpi::experimental::rank                                       Dest,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm>
auto isend(SendMode&& mode, Request&& request, SBuf&& sbuf, Dest dest, Comm const& comm) -> SBuf {
    return kamping::v2::isend(
        mode,
        std::forward<Request>(request),
        std::forward<SBuf>(sbuf),
        std::move(dest),
        DEFAULT_SEND_TAG,
        comm
    );
}

/// Starts sending `sbuf` to `dest` with `tag` in the given @ref send_mode.
/// @return An iresult owning the request (and the buffer, if passed as rvalue); complete it with `wait()` or `test()`.
template <
    is_send_mode                                           SendMode,
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest,
    mpi::experimental::tag                                 Tag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto isend(SendMode&& mode, SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm) -> iresult<SBuf> {
    iresult<SBuf> res{std::forward<SBuf>(sbuf)};
    kamping::v2::isend(mode, res.mpi_handle_ptr(), res.view(), std::move(dest), std::move(tag), comm);
    return res;
}

/// Like the overload above, with @ref DEFAULT_SEND_TAG.
template <
    is_send_mode                                           SendMode,
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto isend(SendMode&& mode, SBuf&& sbuf, Dest dest, Comm const& comm) -> iresult<SBuf> {
    return kamping::v2::isend(mode, std::forward<SBuf>(sbuf), std::move(dest), DEFAULT_SEND_TAG, comm);
}

/// Starts sending `sbuf` to `dest` with `tag` in standard mode, using the caller-supplied `request` (`MPI_Isend`).
/// @return The send buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request,
    mpi::experimental::send_buffer                                SBuf,
    mpi::experimental::rank                                       Dest,
    mpi::experimental::tag                                        Tag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm>
auto isend(Request&& request, SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm) -> SBuf {
    return kamping::v2::isend(
        send_mode::standard,
        std::forward<Request>(request),
        std::forward<SBuf>(sbuf),
        std::move(dest),
        std::move(tag),
        comm
    );
}

/// Like the overload above, with @ref DEFAULT_SEND_TAG.
template <
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request,
    mpi::experimental::send_buffer                                SBuf,
    mpi::experimental::rank                                       Dest,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm>
auto isend(Request&& request, SBuf&& sbuf, Dest dest, Comm const& comm) -> SBuf {
    return kamping::v2::isend(
        send_mode::standard,
        std::forward<Request>(request),
        std::forward<SBuf>(sbuf),
        std::move(dest),
        DEFAULT_SEND_TAG,
        comm
    );
}

/// Starts sending `sbuf` to `dest` with `tag` in standard mode (`MPI_Isend`).
/// @return An iresult owning the request (and the buffer, if passed as rvalue); complete it with `wait()` or `test()`.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest,
    mpi::experimental::tag                                 Tag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto isend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm) -> iresult<SBuf> {
    return kamping::v2::isend(send_mode::standard, std::forward<SBuf>(sbuf), std::move(dest), std::move(tag), comm);
}

/// Like the overload above, with @ref DEFAULT_SEND_TAG.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto isend(SBuf&& sbuf, Dest dest, Comm const& comm) -> iresult<SBuf> {
    return kamping::v2::isend(send_mode::standard, std::forward<SBuf>(sbuf), std::move(dest), DEFAULT_SEND_TAG, comm);
}

} // namespace kamping::v2
