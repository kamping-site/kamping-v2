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
