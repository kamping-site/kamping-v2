// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "mpi/collectives/allgatherv.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {
/// Gathers a variable amount of data from all ranks into the variadic buffer `rbuf` on every rank (`MPI_Allgatherv`).
///
/// If `rbuf` has deferred counts (e.g. `auto_recv_v<T>()`), the counts are exchanged first and `rbuf` is sized to fit.
/// @return The send and receive buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer_v                       RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto allgatherv(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::allgatherv{}, res.send, res.recv, mpi::experimental::handle(comm));
    mpi::experimental::allgatherv(res.send, res.recv, comm);
    return res;
}
} // namespace kamping::v2
