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
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto allgather(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::allgather{}, res.send, res.recv, mpi::experimental::handle(comm));
    if constexpr (mpi::experimental::has_large_count<SBuf> && mpi::experimental::has_large_count<RBuf>) {
#if MPI_VERSION >= 4
        if (count(res.send) > INT_MAX && count(res.recv) > INT_MAX) {
            mpi::experimental::allgather_c(res.send, res.recv, comm);
            return res;
        }
#endif
    }
    mpi::experimental::allgather(res.send, res.recv, comm);
    return res;
}
} // namespace kamping::v2
