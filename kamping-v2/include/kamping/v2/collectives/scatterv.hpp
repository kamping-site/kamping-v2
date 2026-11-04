// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "kamping/v2/sentinels.hpp"
#include "mpi/buffer.hpp"
#include "mpi/collectives/scatterv.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {

/// Distributes parts of the variadic buffer `sbuf` on `root` to all ranks, receiving into `rbuf` (`MPI_Scatterv`).
///
/// The other ranks pass `null_buf_v` as `sbuf` or use the overloads without send buffer. `rbuf` may be a deferred
/// receive buffer (e.g. `views::resize`); root's counts are then scattered first.
/// @return The send and receive buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_buffer_v                       SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::rank                                Root,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto scatterv(SBuf&& sbuf, RBuf&& rbuf, Root root, Comm const& comm) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::scatterv{}, res.send, res.recv, mpi::experimental::to_rank(root), mpi::experimental::handle(comm));
    mpi::experimental::scatterv(res.send, res.recv, std::move(root), comm);
    return res;
}

/// Scatters from rank 0; see the overload above.
template <
    mpi::experimental::send_buffer_v                       SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto scatterv(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm) -> result<SBuf, RBuf> {
    return kamping::v2::scatterv(std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf), 0, comm);
}

/// Non-root shorthand of @ref scatterv: receives this rank's part from `root` into `rbuf`, without a send buffer.
/// @return The receive buffer.
template <
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::rank                                Root,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto scatterv(RBuf&& rbuf, Root root, Comm const& comm) -> RBuf {
    return kamping::v2::scatterv(null_buf_v_t{}, std::forward<RBuf>(rbuf), std::move(root), comm).recv;
}

/// Non-root shorthand of @ref scatterv: receives this rank's part from rank 0 into `rbuf`, without a send buffer.
/// @return The receive buffer.
template <mpi::experimental::recv_buffer RBuf, mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto scatterv(RBuf&& rbuf, Comm const& comm) -> RBuf {
    return kamping::v2::scatterv(null_buf_v_t{}, std::forward<RBuf>(rbuf), 0, comm).recv;
}

} // namespace kamping::v2
