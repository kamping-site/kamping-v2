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
#include "mpi/collectives/gather.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {

/// Gathers `sbuf` from all ranks into `rbuf` on `root` (`MPI_Gather`).
///
/// On root, `rbuf` may be a deferred receive buffer (e.g. `views::resize`); on the other ranks it is ignored, so pass
/// `null_buf` or use the overloads without receive buffer.
/// @return The send and receive buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::rank                                Root,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto gather(SBuf&& sbuf, RBuf&& rbuf, Root root, Comm const& comm) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::gather{}, res.send, res.recv, mpi::experimental::to_rank(root), mpi::experimental::handle(comm));
    mpi::experimental::gather(res.send, res.recv, std::move(root), comm);
    return res;
}

/// Gathers `sbuf` from all ranks into `rbuf` on rank 0; see the overload above.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto gather(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm) -> result<SBuf, RBuf> {
    return kamping::v2::gather(std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf), 0, comm);
}

/// Non-root shorthand of @ref gather: contributes `sbuf` to the gather on `root`, without a receive buffer.
/// @return The send buffer.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Root,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto gather(SBuf&& sbuf, Root root, Comm const& comm) -> SBuf {
    return kamping::v2::gather(std::forward<SBuf>(sbuf), null_buf_t{}, std::move(root), comm).send;
}

/// Non-root shorthand of @ref gather: contributes `sbuf` to the gather on rank 0, without a receive buffer.
/// @return The send buffer.
template <mpi::experimental::send_buffer SBuf, mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto gather(SBuf&& sbuf, Comm const& comm) -> SBuf {
    return kamping::v2::gather(std::forward<SBuf>(sbuf), null_buf_t{}, 0, comm).send;
}

} // namespace kamping::v2
