// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "mpi/collectives/alltoallv.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {
/// Exchanges variable amounts of data between all ranks (`MPI_Alltoallv`).
///
/// `sbuf` is a variadic buffer, e.g. `data | views::with_counts(c) | views::auto_displs()` or `nested |
/// views::flatten_v()`. If `rbuf` has deferred counts (e.g. `auto_recv_v<T>()`), the counts are exchanged first and
/// `rbuf` is sized to fit.
/// @return The send and receive buffer (owned if passed as rvalue, borrowed otherwise).
template <
    mpi::experimental::send_buffer_v                       SBuf,
    mpi::experimental::recv_buffer_v                       RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto alltoallv(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::alltoallv{}, res.send, res.recv, mpi::experimental::handle(comm));
    mpi::experimental::alltoallv(res.send, res.recv, comm);
    return res;
}
} // namespace kamping::v2
