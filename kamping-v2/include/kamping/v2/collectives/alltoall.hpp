// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "mpi/collectives/alltoall.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {
/// Sends an equal share of `sbuf` to every rank and receives one share from every rank into `rbuf` (`MPI_Alltoall`).
///
/// `rbuf` may be a deferred receive buffer (e.g. `views::resize`), which is resized to `count(sbuf)`.
/// @return The send and receive buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto alltoall(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::alltoall{}, res.send, res.recv, mpi::experimental::handle(comm));
    mpi::experimental::alltoall(res.send, res.recv, comm);
    return res;
}
} // namespace kamping::v2
