#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"
#include "mpi/ops.hpp"

namespace mpi::experimental {
template <
    send_buffer                                            SBuf,
    recv_buffer                                            RBuf,
    mpi::experimental::valid_op<SBuf, RBuf>                Op,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void allreduce(SBuf&& sbuf, RBuf&& rbuf, Op const& op, Comm const& comm = MPI_COMM_WORLD) {
    auto sbuf_ptr = ptr(sbuf);
    if (sbuf_ptr == MPI_IN_PLACE) {
        int err = MPI_Allreduce(
            sbuf_ptr,
            ptr(rbuf),
            static_cast<int>(count(rbuf)),
            type(rbuf),
            as_mpi_op(op, sbuf, rbuf),
            handle(comm)
        );
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    } else {
        using scount_t = decltype(count(sbuf));
        KAMPING_ASSERT(
            count(sbuf) == static_cast<scount_t>(count(rbuf)),
            "send and receive buffer must have the same count"
        );
        KAMPING_ASSERT(type(sbuf) == type(rbuf), "send and receive buffer must have the same type");
        int err = MPI_Allreduce(
            sbuf_ptr,
            ptr(rbuf),
            static_cast<int>(count(sbuf)),
            type(sbuf),
            as_mpi_op(op, sbuf, rbuf),
            handle(comm)
        );
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    }
}
} // namespace mpi::experimental
