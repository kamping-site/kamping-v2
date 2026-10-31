// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <type_traits>
#include <utility>

#include <mpi.h>

#include "kamping/v2/p2p/constants.hpp"
#include "kamping/v2/p2p/send_mode.hpp"
#include "mpi/handle.hpp"
#include "mpi/p2p/send.hpp"

namespace kamping::v2 {

/// Sends `sbuf` to `dest` with `tag` in the given @ref send_mode (`MPI_Send`, `MPI_Bsend`, `MPI_Ssend` or
/// `MPI_Rsend`).
/// @return The send buffer (owned if passed as rvalue, borrowed otherwise).
template <
    is_send_mode                                           SendMode,
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest,
    mpi::experimental::tag                                 Tag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto send(SendMode&&, SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm) -> SBuf {
    using mode_t = std::remove_cvref_t<SendMode>;
    if constexpr (std::same_as<mode_t, send_mode::standard_t>) {
        mpi::experimental::send(sbuf, std::move(dest), std::move(tag), comm);
    } else if constexpr (std::same_as<mode_t, send_mode::buffered_t>) {
        mpi::experimental::bsend(sbuf, std::move(dest), std::move(tag), comm);
    } else if constexpr (std::same_as<mode_t, send_mode::sync_t>) {
        mpi::experimental::ssend(sbuf, std::move(dest), std::move(tag), comm);
    } else {
        mpi::experimental::rsend(sbuf, std::move(dest), std::move(tag), comm);
    }
    return std::forward<SBuf>(sbuf);
}

/// Sends `sbuf` to `dest` with @ref DEFAULT_SEND_TAG in the given @ref send_mode.
/// @return The send buffer (owned if passed as rvalue, borrowed otherwise).
template <
    is_send_mode                                           SendMode,
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto send(SendMode&& mode, SBuf&& sbuf, Dest dest, Comm const& comm) -> SBuf {
    return kamping::v2::send(mode, std::forward<SBuf>(sbuf), std::move(dest), DEFAULT_SEND_TAG, comm);
}

/// Sends `sbuf` to `dest` with `tag` in standard mode (`MPI_Send`).
/// @return The send buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest,
    mpi::experimental::tag                                 Tag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto send(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm) -> SBuf {
    return kamping::v2::send(send_mode::standard, std::forward<SBuf>(sbuf), std::move(dest), std::move(tag), comm);
}

/// Sends `sbuf` to `dest` with @ref DEFAULT_SEND_TAG in standard mode (`MPI_Send`).
/// @return The send buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto send(SBuf&& sbuf, Dest dest, Comm const& comm) -> SBuf {
    return kamping::v2::send(send_mode::standard, std::forward<SBuf>(sbuf), std::move(dest), DEFAULT_SEND_TAG, comm);
}

} // namespace kamping::v2
