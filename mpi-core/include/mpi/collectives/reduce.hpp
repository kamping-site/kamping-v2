#pragma once

#include <mpi.h>
#include <mpi/buffer.hpp>
#include <mpi/error.hpp>
#include <mpi/handle.hpp>
#include <mpi/ops.hpp>

namespace mpi::experimental {

/// Reduction operation (blocking).
///
/// Combines elements from the send buffer of each MPI process using the specified operation,
/// and returns the combined value in the receive buffer of the root process.
///
/// @tparam SBuf Send buffer type satisfying send_buffer
/// @tparam RBuf Receive buffer type satisfying recv_buffer
/// @tparam Op Operation type satisfying valid_op<Op, SBuf, RBuf>
/// @tparam Comm Communicator type convertible to MPI_Comm
///
/// @param sbuf Send buffer (or inplace sentinel)
/// @param rbuf Receive buffer
/// @param op Reduction operation (MPI_Op or builtin operation)
/// @param root Rank of the root process
/// @param comm MPI communicator (default: MPI_COMM_WORLD)
///
/// Preconditions:
/// - For non-inplace: on root, `count(sbuf) == count(rbuf)` and `type(sbuf) == type(rbuf)`
/// - For inplace: count and type are derived from rbuf
template <
    send_buffer                         SBuf,
    recv_buffer                         RBuf,
    valid_op<SBuf, RBuf>                Op,
    rank                                Root = int,
    convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void reduce(SBuf&& sbuf, RBuf&& rbuf, Op const& op, Root root, Comm const& comm = MPI_COMM_WORLD) {
    auto sbuf_ptr = ptr(sbuf);
    int  rank     = 0;
    MPI_Comm_rank(handle(comm), &rank);
    int root_rank = to_rank(root);

    if (sbuf_ptr == MPI_IN_PLACE) {
        // Inplace: count and type from rbuf
        KAMPING_ASSERT(rank == root_rank, "inplace reduce only valid on root");
        int err = MPI_Reduce(
            sbuf_ptr,
            ptr(rbuf),
            static_cast<int>(count(rbuf)),
            type(rbuf),
            as_mpi_op(op, sbuf, rbuf),
            root_rank,
            handle(comm)
        );
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    } else {
        // Normal: count and type from sbuf
        using scount_t = decltype(count(sbuf));
        KAMPING_ASSERT(
            rank != root_rank || count(sbuf) == static_cast<scount_t>(count(rbuf)),
            "on root: send and receive buffers must have the same count"
        );
        KAMPING_ASSERT(
            rank != root_rank || type(sbuf) == type(rbuf),
            "on root: send and receive buffers must have the same type"
        );
        int err = MPI_Reduce(
            sbuf_ptr,
            ptr(rbuf),
            static_cast<int>(count(sbuf)),
            type(sbuf),
            as_mpi_op(op, sbuf, rbuf),
            root_rank,
            handle(comm)
        );
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    }
}

} // namespace mpi::experimental
