#pragma once

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "kamping/v2/sentinels.hpp"
#include "mpi/buffer.hpp"
#include "mpi/collectives/gather.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::rank                                Root = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto gather(SBuf&& sbuf, RBuf&& rbuf, Root root = 0, Comm const& comm = MPI_COMM_WORLD) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::gather{}, res.send, res.recv, mpi::experimental::to_rank(root), mpi::experimental::handle(comm));
    mpi::experimental::gather(res.send, res.recv, std::move(root), comm);
    return res;
}

template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Root = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto gather(SBuf&& sbuf, Root root = 0, Comm const& comm = MPI_COMM_WORLD) -> SBuf {
    auto res = kamping::v2::gather(std::forward<SBuf>(sbuf), null_buf_t{}, std::move(root), comm);
    return std::move(res).send;
}
} // namespace kamping::v2
