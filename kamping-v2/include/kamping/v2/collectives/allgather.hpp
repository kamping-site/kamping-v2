// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "mpi/collectives/allgather.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {
/// Gathers `sbuf` from all ranks into `rbuf` on every rank (`MPI_Allgather`).
///
/// `rbuf` may be a deferred receive buffer (e.g. `views::resize`), which is resized to `comm_size * count(sbuf)`.
/// @return The send and receive buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
auto allgather(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::allgather{}, res.send, res.recv, mpi::experimental::handle(comm));
    mpi::experimental::allgather(res.send, res.recv, comm);
    return res;
}
} // namespace kamping::v2
