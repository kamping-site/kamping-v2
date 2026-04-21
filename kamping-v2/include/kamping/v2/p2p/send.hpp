#pragma once

#include <mpi.h>

#include "kamping/v2/p2p/constants.hpp"
#include "kamping/v2/p2p/send_mode.hpp"
#include "mpi/handle.hpp"
#include "mpi/p2p/send.hpp"

namespace kamping::v2 {

template <
    is_send_mode                                           SendMode,
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest = int,
    mpi::experimental::tag                                 Tag  = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto send(SendMode&&, SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD) -> SBuf {
    auto send_impl = [](auto&&... args) {
        if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::standard_t>) {
            mpi::experimental::send(std::forward<decltype(args)>(args)...);
        } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::buffered_t>) {
            mpi::experimental::bsend(std::forward<decltype(args)>(args)...);
        } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::sync_t>) {
            mpi::experimental::ssend(std::forward<decltype(args)>(args)...);
        } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::ready_t>) {
            mpi::experimental::rsend(std::forward<decltype(args)>(args)...);
        }
    };
    send_impl(sbuf, std::move(dest), std::move(tag), comm);
    return std::forward<SBuf>(sbuf);
}

template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest = int,
    mpi::experimental::tag                                 Tag  = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto send(SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD) -> SBuf {
    return send(send_mode::standard, std::forward<SBuf>(sbuf), std::move(dest), std::move(tag), comm);
}

template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto send(SBuf&& sbuf, Dest dest, Comm const& comm) -> SBuf {
    return send(send_mode::standard, std::forward<SBuf>(sbuf), std::move(dest), DEFAULT_SEND_TAG, comm);
}
} // namespace kamping::v2
