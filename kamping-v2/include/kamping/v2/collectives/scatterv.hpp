#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "kamping/v2/sentinels.hpp"
#include "mpi/buffer.hpp"
#include "mpi/collectives/scatterv.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {

/// Scatterv (blocking).
///
/// Distributes variable-length data from root to all ranks. All ranks must call
/// this function. On non-root, pass `v2::null_buf_v` as sbuf (MPI ignores it).
///
/// @param sbuf  Variadic send buffer on root (with per-rank counts and displacements);
///              `null_buf_v` on non-root.
/// @param rbuf  Recv buffer on every rank; may be a resizable view (`views::resize`)
///              to trigger automatic count inference via MPI_Scatter of root's sendcounts.
/// @param root  Root rank (default: 0).
/// @param comm  MPI communicator (default: MPI_COMM_WORLD).
template <
    mpi::experimental::send_buffer_v                       SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::rank                                Root = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto scatterv(SBuf&& sbuf, RBuf&& rbuf, Root root = 0, Comm const& comm = MPI_COMM_WORLD) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::scatterv{}, res.send, res.recv, mpi::experimental::to_rank(root), mpi::experimental::handle(comm));
    mpi::experimental::scatterv(res.send, res.recv, std::move(root), comm);
    return res;
}

} // namespace kamping::v2
