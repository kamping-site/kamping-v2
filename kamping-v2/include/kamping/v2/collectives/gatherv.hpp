#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "mpi/collectives/gatherv.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {

/// Gatherv (blocking).
///
/// Gathers variable-length data from all ranks to root. All ranks must call
/// this function. Non-root ranks pass `v2::auto_null_recv_v` (to trigger
/// automatic count inference on root) or `v2::null_buf_v` (when root supplies
/// counts explicitly).
///
/// @param sbuf  Send buffer on every rank.
/// @param rbuf  Recv buffer on root (e.g. `auto_recv_v<T>()` for auto-resize);
///              `auto_null_recv_v` on non-root to participate in count inference,
///              or `null_buf_v` when counts are already known.
/// @param root  Root rank.
/// @param comm  MPI communicator (default: MPI_COMM_WORLD).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer_v                       RBuf,
    mpi::experimental::rank                                Root = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto gatherv(SBuf&& sbuf, RBuf&& rbuf, Root root = 0, Comm const& comm = MPI_COMM_WORLD) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::gatherv{}, res.send, res.recv, mpi::experimental::to_rank(root), mpi::experimental::handle(comm));
    mpi::experimental::gatherv(res.send, res.recv, std::move(root), comm);
    return res;
}

} // namespace kamping::v2
