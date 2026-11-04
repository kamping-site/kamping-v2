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

/// @file
/// Reduction to a root. The operation defaults to `std::plus<>` and the root to rank 0; a root can only be given
/// together with an operation, since on some MPI implementations `MPI_Op` and ranks are both `int`.

namespace kamping::v2 {

/// Reduces `sbuf` across all ranks with `op` and stores the result in `rbuf` on `root` (`MPI_Reduce`).
///
/// Pass `inplace` as `sbuf` on root to reduce `rbuf` in place. On root, `rbuf` may be a deferred receive buffer (e.g.
/// `views::resize`); on the other ranks it is ignored, so pass `null_buf` or use the overloads without receive buffer.
/// @param op Reduction operation: an `MPI_Op`, a standard functor such as `std::plus<>`, or the result of `make_op`.
/// @return The send and receive buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::valid_op<SBuf, RBuf>                Op,
    mpi::experimental::rank                                Root,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto reduce(SBuf&& sbuf, RBuf&& rbuf, Op const& op, Root root, Comm const& comm) -> result<SBuf, RBuf> {
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

/// Reduces with `op` to rank 0; see the overload above.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::valid_op<SBuf, RBuf>                Op,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto reduce(SBuf&& sbuf, RBuf&& rbuf, Op const& op, Comm const& comm) -> result<SBuf, RBuf> {
    return kamping::v2::reduce(std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf), op, 0, comm);
}

/// Sums to rank 0; see the overload above.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
    requires mpi::experimental::valid_op<std::plus<>, SBuf, RBuf>
auto reduce(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm) -> result<SBuf, RBuf> {
    return kamping::v2::reduce(std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf), std::plus<>{}, 0, comm);
}

/// Non-root shorthand of @ref reduce: contributes `sbuf` to the reduction with `op` on `root`, without a receive
/// buffer.
/// @return The send buffer.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::valid_op<SBuf, null_buf_t>          Op,
    mpi::experimental::rank                                Root,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto reduce(SBuf&& sbuf, Op const& op, Root root, Comm const& comm) -> SBuf {
    return kamping::v2::reduce(std::forward<SBuf>(sbuf), null_buf_t{}, op, std::move(root), comm).send;
}

/// Non-root shorthand of @ref reduce: contributes `sbuf` to the reduction with `op` on rank 0.
/// @return The send buffer.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::valid_op<SBuf, null_buf_t>          Op,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto reduce(SBuf&& sbuf, Op const& op, Comm const& comm) -> SBuf {
    return kamping::v2::reduce(std::forward<SBuf>(sbuf), null_buf_t{}, op, 0, comm).send;
}

/// Non-root shorthand of @ref reduce: contributes `sbuf` to the sum on rank 0.
/// @return The send buffer.
template <mpi::experimental::send_buffer SBuf, mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
    requires mpi::experimental::valid_op<std::plus<>, SBuf, null_buf_t>
auto reduce(SBuf&& sbuf, Comm const& comm) -> SBuf {
    return kamping::v2::reduce(std::forward<SBuf>(sbuf), null_buf_t{}, std::plus<>{}, 0, comm).send;
}

} // namespace kamping::v2
