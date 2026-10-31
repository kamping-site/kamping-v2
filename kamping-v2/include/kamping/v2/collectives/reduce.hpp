// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <functional>
#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "kamping/v2/sentinels.hpp"
#include "mpi/collectives/reduce.hpp"
#include "mpi/handle.hpp"
#include "mpi/ops.hpp"

namespace kamping::v2 {

/// Reduces `sbuf` across all ranks with `op` and stores the result in `rbuf` on `root` (`MPI_Reduce`).
///
/// Pass `inplace` as `sbuf` on root to reduce `rbuf` in place. On root, `rbuf` may be a deferred receive buffer (e.g.
/// `views::resize`); on the other ranks it is ignored.
/// @param op Reduction operation: an `MPI_Op`, a standard functor such as `std::plus<>` (the default), or the result of
/// `make_op`.
/// @return The send and receive buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::valid_op<SBuf, RBuf>                Op   = std::plus<>,
    mpi::experimental::rank                                Root = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto reduce(SBuf&& sbuf, RBuf&& rbuf, Op const& op = std::plus<>{}, Root root = 0, Comm const& comm = MPI_COMM_WORLD)
    -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(
        comm_op::reduce{},
        res.send,
        res.recv,
        mpi::experimental::as_mpi_op(op, res.send, res.recv),
        mpi::experimental::to_rank(root),
        mpi::experimental::handle(comm)
    );
    mpi::experimental::reduce(res.send, res.recv, op, std::move(root), comm);
    return res;
}

/// Non-root shorthand of @ref reduce: contributes `sbuf` without a receive buffer.
/// @return The send buffer.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::valid_op<SBuf, null_buf_t>          Op   = std::plus<>,
    mpi::experimental::rank                                Root = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto reduce(SBuf&& sbuf, Op const& op = std::plus<>{}, Root root = 0, Comm const& comm = MPI_COMM_WORLD) -> SBuf {
    auto res = kamping::v2::reduce(std::forward<SBuf>(sbuf), null_buf_t{}, op, std::move(root), comm);
    return std::move(res).send;
}

} // namespace kamping::v2
