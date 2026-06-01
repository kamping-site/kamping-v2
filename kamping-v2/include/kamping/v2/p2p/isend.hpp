// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "kamping/v2/iresult.hpp"
#include "kamping/v2/p2p/constants.hpp"
#include "kamping/v2/p2p/send_mode.hpp"
#include "mpi/handle.hpp"
#include "mpi/p2p/isend.hpp"

namespace kamping::v2 {

/// Low-level overload: caller supplies an external MPI_Request* and manages its lifetime.
/// Returns the buffer (ownership semantics match blocking send).
template <
    is_send_mode                                                  SendMode,
    mpi::experimental::send_buffer                                SBuf,
    mpi::experimental::rank                                       Dest    = int,
    mpi::experimental::tag                                        Tag     = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
auto isend(
    SendMode&&, Request&& request, SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD
) -> SBuf {
    if constexpr (mpi::experimental::has_large_count<SBuf>) {
#if MPI_VERSION >= 4
        if (count(sbuf) > INT_MAX) {
            if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::standard_t>) {
                mpi::experimental::isend_c(sbuf, std::move(dest), std::move(tag), comm, std::forward<Request>(request));
            } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::buffered_t>) {
                mpi::experimental::ibsend_c(sbuf, std::move(dest), std::move(tag), comm, std::forward<Request>(request));
            } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::sync_t>) {
                mpi::experimental::issend_c(sbuf, std::move(dest), std::move(tag), comm, std::forward<Request>(request));
            } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::ready_t>) {
                mpi::experimental::irsend_c(sbuf, std::move(dest), std::move(tag), comm, std::forward<Request>(request));
            }
            return std::forward<SBuf>(sbuf);
        }
#endif
    }
    if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::standard_t>) {
        mpi::experimental::isend(sbuf, std::move(dest), std::move(tag), comm, std::forward<Request>(request));
    } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::buffered_t>) {
        mpi::experimental::ibsend(sbuf, std::move(dest), std::move(tag), comm, std::forward<Request>(request));
    } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::sync_t>) {
        mpi::experimental::issend(sbuf, std::move(dest), std::move(tag), comm, std::forward<Request>(request));
    } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::ready_t>) {
        mpi::experimental::irsend(sbuf, std::move(dest), std::move(tag), comm, std::forward<Request>(request));
    }
    return std::forward<SBuf>(sbuf);
}
/// High-level overload: creates and owns the MPI_Request internally.
/// Returns iresult<SBuf> which co-locates the request and buffer;
/// call wait() to block and retrieve the buffer.
template <
    is_send_mode                                           SendMode,
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest = int,
    mpi::experimental::tag                                 Tag  = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto isend(SendMode&&, SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD)
    -> iresult<SBuf> {
    iresult<SBuf> res{std::forward<SBuf>(sbuf)};
    if constexpr (mpi::experimental::has_large_count<SBuf>) {
#if MPI_VERSION >= 4
        if (count(res.view()) > INT_MAX) {
            if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::standard_t>) {
                mpi::experimental::isend_c(res.view(), dest, tag, comm, res.mpi_native_handle_ptr());
            } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::buffered_t>) {
                mpi::experimental::ibsend_c(res.view(), dest, tag, comm, res.mpi_native_handle_ptr());
            } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::sync_t>) {
                mpi::experimental::issend_c(res.view(), dest, tag, comm, res.mpi_native_handle_ptr());
            } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::ready_t>) {
                mpi::experimental::irsend_c(res.view(), dest, tag, comm, res.mpi_native_handle_ptr());
            }
            return res;
        }
#endif
    }
    if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::standard_t>) {
        mpi::experimental::isend(res.view(), dest, tag, comm, res.mpi_native_handle_ptr());
    } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::buffered_t>) {
        mpi::experimental::ibsend(res.view(), dest, tag, comm, res.mpi_native_handle_ptr());
    } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::sync_t>) {
        mpi::experimental::issend(res.view(), dest, tag, comm, res.mpi_native_handle_ptr());
    } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::ready_t>) {
        mpi::experimental::irsend(res.view(), dest, tag, comm, res.mpi_native_handle_ptr());
    }
    return res;
}

/// Convenience overload: standard mode, external request.
template <
    mpi::experimental::send_buffer                                SBuf,
    mpi::experimental::rank                                       Dest    = int,
    mpi::experimental::tag                                        Tag     = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
auto isend(Request&& request, SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD)
    -> SBuf {
    return isend(
        send_mode::standard,
        std::forward<Request>(request),
        std::forward<SBuf>(sbuf),
        std::move(dest),
        std::move(tag),
        comm
    );
}

/// Convenience overload: standard mode, internally managed request (returns iresult).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest = int,
    mpi::experimental::tag                                 Tag  = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto isend(SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD) -> iresult<SBuf> {
    return kamping::v2::isend(send_mode::standard, std::forward<SBuf>(sbuf), std::move(dest), std::move(tag), comm);
}

/// Convenience overload: standard mode, no explicit tag, comm passed directly.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto isend(SBuf&& sbuf, Dest dest, Comm const& comm) -> iresult<SBuf> {
    return kamping::v2::isend(send_mode::standard, std::forward<SBuf>(sbuf), std::move(dest), DEFAULT_SEND_TAG, comm);
}

} // namespace kamping::v2
