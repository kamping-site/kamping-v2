// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/infer.hpp"
#include "mpi/collectives/bcast.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {
/// Broadcasts `send_recv_buf` from `root` to all ranks (`MPI_Bcast`).
///
/// If the buffer is a deferred receive buffer (e.g. `views::resize`), root's count is broadcast first and the buffer is
/// resized on the other ranks.
/// @return The buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_recv_buffer                    SRBuf,
    mpi::experimental::rank                                Root = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto bcast(SRBuf&& send_recv_buf, Root root = 0, Comm const& comm = MPI_COMM_WORLD) -> SRBuf {
    infer(comm_op::bcast{}, send_recv_buf, mpi::experimental::to_rank(root), mpi::experimental::handle(comm));
    mpi::experimental::bcast(send_recv_buf, std::move(root), comm);
    return std::forward<SRBuf>(send_recv_buf);
}
} // namespace kamping::v2
