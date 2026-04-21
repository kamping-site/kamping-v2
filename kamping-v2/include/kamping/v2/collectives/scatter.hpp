#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "kamping/v2/sentinels.hpp"
#include "mpi/buffer.hpp"
#include "mpi/collectives/scatter.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {

/// Scatter (blocking, two-buffer form).
///
/// Distributes data from root to all ranks. All ranks must call this function.
/// On non-root, pass `v2::null_buf` as sbuf (MPI ignores it). On root, rbuf
/// may be `v2::inplace` to keep root's own segment in place within sbuf.
///
/// @param sbuf  Send buffer on root (total count must be divisible by comm size);
///              `null_buf` on non-root.
/// @param rbuf  Recv buffer on every rank; may be a resizable view (`views::resize`)
///              or `v2::inplace` on root.
/// @param root  Root rank (default: 0).
/// @param comm  MPI communicator (default: MPI_COMM_WORLD).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::rank                                Root = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto scatter(SBuf&& sbuf, RBuf&& rbuf, Root root = 0, Comm const& comm = MPI_COMM_WORLD) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::scatter{}, res.send, res.recv, mpi::experimental::to_rank(root), mpi::experimental::handle(comm));
    mpi::experimental::scatter(res.send, res.recv, std::move(root), comm);
    return res;
}

/// Scatter (blocking, non-root shorthand).
///
/// Convenience overload for non-root ranks: sbuf defaults to `null_buf`.
/// Root must use the two-buffer form.
///
/// @param rbuf  Recv buffer; may be a resizable view (`views::resize`).
/// @param root  Root rank (default: 0).
/// @param comm  MPI communicator (default: MPI_COMM_WORLD).
template <
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::rank                                Root = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto scatter(RBuf&& rbuf, Root root = 0, Comm const& comm = MPI_COMM_WORLD) -> RBuf {
    auto res = kamping::v2::scatter(null_buf_t{}, std::forward<RBuf>(rbuf), std::move(root), comm);
    return std::move(res).recv;
}

} // namespace kamping::v2
