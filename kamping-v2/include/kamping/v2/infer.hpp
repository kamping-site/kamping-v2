// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <cstdio>
#include <exception>
#include <optional>
#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/p2p/probe.hpp"
#include "kamping/v2/status.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "mpi/collectives/allgather.hpp"
#include "mpi/collectives/alltoall.hpp"
#include "mpi/collectives/bcast.hpp"
#include "mpi/collectives/gather.hpp"
#include "mpi/collectives/scatter.hpp"
#include "mpi/comm.hpp"
#include "mpi/p2p/probe.hpp"
#include "mpi/p2p/sendrecv.hpp"

/// @file
/// infer() is a customization point that transfers metadata from the sending to the receiving
/// side before an MPI operation is issued. The default overloads below size deferred receive
/// buffers (see deferred_recv_buf / deferred_recv_buf_v) and are no-ops for all other buffers.
/// Users can provide their own overloads via ADL for custom buffer types or to transfer
/// additional metadata alongside the count.
///
/// Dispatch is on operation tag types (comm_op::recv, comm_op::allgather, ...) rather than an enum, so
/// users can add new tags without modifying this header.

namespace kamping::v2 {

namespace detail {
/// Returns the probed element count, or terminates if the message is not a whole number of elements of the receive
/// buffer's datatype (MPI_Get_count yields MPI_UNDEFINED).
///
/// Checked in all build types, since the count depends on what the remote rank sent. Terminates rather than throws
/// because a message matched by MPI_Mprobe cannot be dropped (see mpi::experimental::message) and a mismatch between
/// the sent and received datatype is a programming error.
inline std::ptrdiff_t checked_recv_count(std::optional<int> count) noexcept {
    if (!count) {
        std::fprintf(
            stderr,
            "kamping: probed message is not a whole number of elements of the receive buffer's datatype "
            "(MPI_Get_count returned MPI_UNDEFINED). Check that sender and receiver use matching datatypes.\n"
        );
        std::fflush(stderr);
        std::terminate();
    }
    return static_cast<std::ptrdiff_t>(*count);
}
} // namespace detail

// ---- Default infer() overloads ----------------------------------------------

/// recv / irecv: probes the message (`MPI_Mprobe`, or `MPI_Probe` if the buffer does not support matched probes) to
/// size a deferred receive buffer. Returns the matched `MPI_Message`, or the resolved `{source, tag}` after a plain
/// probe, which the caller then receives.
template <mpi::experimental::recv_buffer RBuf>
auto infer(comm_op::recv, RBuf& rbuf, int source, int tag, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        if constexpr (!kamping::v2::supports_matched_probe<RBuf>) {
            // Matched recv is disabled for this buffer type (e.g. GPU buffers with
            // MPI implementations that don't support GPU-aware MPI_Mrecv).
            // Use a plain MPI_Probe and return the resolved {source, tag} so recv.hpp
            // can issue a targeted MPI_Recv rather than a wildcard one.
            v2::status status;
            mpi::experimental::probe(source, tag, comm, status);
            rbuf.set_recv_count(v2::detail::checked_recv_count(status.count(mpi::experimental::type(rbuf))));
            return std::pair{status.source(), status.tag()};
        } else {
            auto res = v2::mprobe(source, tag, comm);
            rbuf.set_recv_count(v2::detail::checked_recv_count(res.count(mpi::experimental::type(rbuf))));
            return res.release();
        }
    }
}

/// mrecv / imrecv: the message has already been probed, so a deferred receive buffer is sized from the probed count.
template <mpi::experimental::recv_buffer RBuf>
void infer(comm_op::mrecv, RBuf& rbuf, v2::probe_result const& probe) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        rbuf.set_recv_count(v2::detail::checked_recv_count(probe.count(mpi::experimental::type(rbuf))));
    }
}

/// bcast: broadcasts root's count to size a deferred buffer on the other ranks.
template <mpi::experimental::send_recv_buffer SRBuf>
auto infer(comm_op::bcast, SRBuf& srbuf, int root, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf<SRBuf>) {
        mpi::experimental::comm_view cv{comm};
        int send_count = cv.rank() == root ? static_cast<int>(mpi::experimental::count(srbuf)) : 0;
        mpi::experimental::bcast(v2::views::ref_single(send_count), root, comm);
        if (cv.rank() != root) {
            srbuf.set_recv_count(static_cast<std::ptrdiff_t>(send_count));
        }
    }
}

/// allgather: sizes a deferred receive buffer to `comm_size * count(sbuf)`.
template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::allgather, SBuf const& sbuf, RBuf& rbuf, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        int comm_size = mpi::experimental::comm_view{comm}.size();
        rbuf.set_recv_count(comm_size * static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
    }
}

/// allgatherv: allgathers the send counts into the deferred counts of the receive buffer.
template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer_v RBuf>
void infer(comm_op::allgatherv, SBuf const& sbuf, RBuf& rbuf, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf_v<RBuf>) {
        int comm_size  = mpi::experimental::comm_view{comm}.size();
        int send_count = static_cast<int>(mpi::experimental::count(sbuf));
        rbuf.set_comm_size(comm_size);
        mpi::experimental::allgather(v2::views::ref_single(send_count), mpi::experimental::counts(rbuf), comm);
        rbuf.commit_counts();
    }
}

/// alltoall: sizes a deferred receive buffer to `count(sbuf)`.
template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::alltoall, SBuf const& sbuf, RBuf& rbuf, MPI_Comm /* comm */) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
    }
}

/// alltoallv: tells a deferred send buffer (e.g. flatten_v_view) the communicator size so it can lay out its counts,
/// then alltoalls the send counts into the deferred counts of the receive buffer. `sbuf` is non-const for the former.
template <mpi::experimental::send_buffer_v SBuf, mpi::experimental::recv_buffer_v RBuf>
void infer(comm_op::alltoallv, SBuf& sbuf, RBuf& rbuf, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_send_buf_v<SBuf> || kamping::v2::deferred_recv_buf_v<RBuf>) {
        int comm_size = mpi::experimental::comm_view{comm}.size();
        if constexpr (kamping::v2::deferred_send_buf_v<SBuf>) {
            sbuf.set_comm_size(comm_size);
        }
        if constexpr (kamping::v2::deferred_recv_buf_v<RBuf>) {
            rbuf.set_comm_size(comm_size);
            mpi::experimental::alltoall(mpi::experimental::counts(sbuf), mpi::experimental::counts(rbuf), comm);
            rbuf.commit_counts();
        }
    }
}

/// sendrecv / isendrecv: exchanges the send counts to size a deferred receive buffer.
template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(
    comm_op::sendrecv, SBuf const& sbuf, RBuf& rbuf, int dest, int send_tag, int source, int recv_tag, MPI_Comm comm
) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        int send_count = static_cast<int>(mpi::experimental::count(sbuf));
        int recv_count = 0;
        mpi::experimental::sendrecv(
            v2::views::ref_single(send_count),
            dest,
            send_tag,
            v2::views::ref_single(recv_count),
            source,
            recv_tag,
            comm,
            MPI_STATUS_IGNORE
        );
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(recv_count));
    }
}

/// reduce: sizes a deferred receive buffer on root to `count(sbuf)`.
template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::reduce, SBuf const& sbuf, RBuf& rbuf, MPI_Op, int root, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        if (mpi::experimental::comm_view{comm}.rank() == root && mpi::experimental::ptr(sbuf) != MPI_IN_PLACE) {
            rbuf.set_recv_count(static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
        }
    }
}

/// allreduce: sizes a deferred receive buffer to `count(sbuf)`.
template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::allreduce, SBuf const& sbuf, RBuf& rbuf, MPI_Op, MPI_Comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        if (mpi::experimental::ptr(sbuf) != MPI_IN_PLACE) {
            rbuf.set_recv_count(static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
        }
    }
}

/// gather: sizes a deferred receive buffer on root to `comm_size * count(sbuf)`.
template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::gather, SBuf const& sbuf, RBuf& rbuf, int root, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        mpi::experimental::comm_view cv{comm};
        if (cv.rank() == root) {
            rbuf.set_recv_count(cv.size() * static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
        }
    }
}

/// gatherv: gathers the send counts into the deferred counts of the receive buffer on root.
template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer_v RBuf>
void infer(comm_op::gatherv, SBuf const& sbuf, RBuf& rbuf, int root, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf_v<RBuf>) {
        mpi::experimental::comm_view cv{comm};
        int                          my_count = static_cast<int>(mpi::experimental::count(sbuf));
        if (cv.rank() == root) {
            rbuf.set_comm_size(cv.size());
        }
        mpi::experimental::gather(v2::views::ref_single(my_count), mpi::experimental::counts(rbuf), root, comm);
        if (cv.rank() == root) {
            rbuf.commit_counts();
        }
    }
}

/// scatter: broadcasts root's per-rank count to size a deferred receive buffer.
template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::scatter, SBuf const& sbuf, RBuf& rbuf, int root, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        mpi::experimental::comm_view cv{comm};
        int                          per_rank_count = static_cast<int>(mpi::experimental::count(sbuf)) / cv.size();
        mpi::experimental::bcast(v2::views::ref_single(per_rank_count), root, comm);
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(per_rank_count));
    }
}

/// scatterv: tells a deferred send buffer on root the communicator size, then scatters root's send counts to size a
/// deferred receive buffer. `sbuf` is non-const for the former; non-root ranks pass `null_buf_v`.
template <mpi::experimental::send_buffer_v SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::scatterv, SBuf& sbuf, RBuf& rbuf, int root, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_send_buf_v<SBuf>) {
        mpi::experimental::comm_view cv{comm};
        if (cv.rank() == root) {
            sbuf.set_comm_size(cv.size());
        }
    }
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        int recv_count = 0;
        mpi::experimental::scatter(mpi::experimental::counts(sbuf), v2::views::ref_single(recv_count), root, comm);
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(recv_count));
    }
}

/// scan: sizes a deferred receive buffer to `count(sbuf)`.
template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::scan, SBuf const& sbuf, RBuf& rbuf, MPI_Op, MPI_Comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        if (mpi::experimental::ptr(sbuf) != MPI_IN_PLACE) {
            rbuf.set_recv_count(static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
        }
    }
}

/// exscan: sizes a deferred receive buffer to `count(sbuf)`.
template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::exscan, SBuf const& sbuf, RBuf& rbuf, MPI_Op, MPI_Comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        if (mpi::experimental::ptr(sbuf) != MPI_IN_PLACE) {
            rbuf.set_recv_count(static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
        }
    }
}

} // namespace kamping::v2
