#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "mpi/collectives/alltoallv.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {
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
