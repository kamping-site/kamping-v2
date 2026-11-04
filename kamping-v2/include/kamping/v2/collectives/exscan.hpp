// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <functional>
#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "mpi/collectives/exscan.hpp"
#include "mpi/handle.hpp"
#include "mpi/ops.hpp"

namespace kamping::v2 {

/// Computes the exclusive prefix reduction of `sbuf` over the ranks with `op` into `rbuf` (`MPI_Exscan`).
///
/// Pass `inplace` as `sbuf` to use `rbuf` for both input and output. `rbuf` may be a deferred receive buffer (e.g.
/// `views::resize`).
///
/// The content of `rbuf` on rank 0 is undefined after the call, as in MPI.
/// @param op Reduction operation: an `MPI_Op`, a standard functor such as `std::plus<>`, or the result of `make_op`.
/// @return The send and receive buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::valid_op<SBuf, RBuf>                Op,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto exscan(SBuf&& sbuf, RBuf&& rbuf, Op const& op, Comm const& comm) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(
        comm_op::exscan{},
        res.send,
        res.recv,
        mpi::experimental::as_mpi_op(op, res.send, res.recv),
        mpi::experimental::handle(comm)
    );
    mpi::experimental::exscan(res.send, res.recv, op, comm);
    return res;
}

/// Like the overload above, with `std::plus<>` as operation.
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
    requires mpi::experimental::valid_op<std::plus<>, SBuf, RBuf>
auto exscan(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm) -> result<SBuf, RBuf> {
    return kamping::v2::exscan(std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf), std::plus<>{}, comm);
}

} // namespace kamping::v2
