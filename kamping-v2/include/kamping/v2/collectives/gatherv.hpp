// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "mpi/collectives/gatherv.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {

/// Gathers variable amounts of data from all ranks into the variadic buffer `rbuf` on `root` (`MPI_Gatherv`).
///
/// On root, `rbuf` may have deferred counts (e.g. `auto_recv_v<T>()`); the counts are then gathered first. The other
/// ranks must then pass `auto_null_recv_v()` to take part in that exchange, or `null_buf_v` if root's counts are
/// given explicitly. There is no non-root shorthand, because which of the two is right depends on root.
/// @return The send and receive buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer_v                       RBuf,
    mpi::experimental::rank                                Root,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto gatherv(SBuf&& sbuf, RBuf&& rbuf, Root root, Comm const& comm) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::gatherv{}, res.send, res.recv, mpi::experimental::to_rank(root), mpi::experimental::handle(comm));
    mpi::experimental::gatherv(res.send, res.recv, std::move(root), comm);
    return res;
}

/// Gathers into `rbuf` on rank 0; see the overload above.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer_v                       RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto gatherv(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm) -> result<SBuf, RBuf> {
    return kamping::v2::gatherv(std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf), 0, comm);
}

} // namespace kamping::v2
