#pragma once

#include <type_traits>
#include <utility>

#include <mpi.h>

#include "kamping/v2/infer.hpp"
#include "mpi/handle.hpp"
#include "mpi/p2p/mrecv.hpp"
#include "mpi/p2p/recv.hpp"

namespace kamping::v2 {
template <
    mpi::experimental::recv_buffer                               RBuf,
    mpi::experimental::rank                                      Source = int,
    mpi::experimental::tag                                       Tag    = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm   = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto recv(
    RBuf&&      rbuf,
    Source      source = MPI_ANY_SOURCE,
    Tag         tag    = MPI_ANY_TAG,
    Comm const& comm   = MPI_COMM_WORLD,
    Status&&    status = MPI_STATUS_IGNORE
) -> RBuf {
    using infer_result_t = decltype(infer(
        comm_op::recv{},
        std::declval<RBuf&>(),
        mpi::experimental::to_rank(std::declval<Source>()),
        mpi::experimental::to_tag(std::declval<Tag>()),
        mpi::experimental::handle(std::declval<Comm const&>())
    ));
    if constexpr (std::is_same_v<infer_result_t, MPI_Message>) {
        MPI_Message msg = infer(
            comm_op::recv{},
            rbuf,
            mpi::experimental::to_rank(source),
            mpi::experimental::to_tag(tag),
            mpi::experimental::handle(comm)
        );
        mpi::experimental::mrecv(rbuf, &msg, std::forward<Status>(status));
        return std::forward<RBuf>(rbuf);
    } else if constexpr (std::is_same_v<infer_result_t, std::pair<int, int>>) {
        // use_matched_probe is false: plain MPI_Probe was used; use resolved source/tag
        // for MPI_Recv to avoid a race with a wildcard receive.
        auto [src, tg] = infer(
            comm_op::recv{},
            rbuf,
            mpi::experimental::to_rank(source),
            mpi::experimental::to_tag(tag),
            mpi::experimental::handle(comm)
        );
        mpi::experimental::recv(rbuf, src, tg, comm, std::forward<Status>(status));
        return std::forward<RBuf>(rbuf);
    } else {
        infer(
            comm_op::recv{},
            rbuf,
            mpi::experimental::to_rank(source),
            mpi::experimental::to_tag(tag),
            mpi::experimental::handle(comm)
        );
        mpi::experimental::recv(rbuf, std::move(source), std::move(tag), comm, std::forward<Status>(status));
        return std::forward<RBuf>(rbuf);
    }
}
template <
    mpi::experimental::recv_buffer                               RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto recv(RBuf&& rbuf, Comm const& comm, Status&& status = MPI_STATUS_IGNORE) -> RBuf {
    return kamping::v2::recv(std::forward<RBuf>(rbuf), MPI_ANY_SOURCE, MPI_ANY_TAG, comm, std::forward<Status>(status));
}
} // namespace kamping::v2
