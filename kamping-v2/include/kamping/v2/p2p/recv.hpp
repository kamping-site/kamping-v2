#pragma once

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
    constexpr bool infer_returns_mpi_message = requires(RBuf&& rbuf_, Source source_, Tag tag_, Comm const& comm_) {
        {
            infer(
                comm_op::recv{},
                rbuf_,
                mpi::experimental::to_rank(source_),
                mpi::experimental::to_tag(tag_),
                mpi::experimental::handle(comm_)
            )
        } -> std::same_as<MPI_Message>;
    };
    if constexpr (infer_returns_mpi_message) {
        MPI_Message msg = infer(
            comm_op::recv{},
            rbuf,
            mpi::experimental::to_rank(source),
            mpi::experimental::to_tag(tag),
            mpi::experimental::handle(comm)
        );
        mpi::experimental::mrecv(rbuf, &msg, std::forward<Status>(status));
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
