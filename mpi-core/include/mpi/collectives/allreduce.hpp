// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"
#include "mpi/ops.hpp"

namespace mpi::experimental {
/// Reduces `sbuf` across all ranks with `op` and stores the result in `rbuf` on every rank (`MPI_Allreduce`).
///
/// Pass `inplace` as `sbuf` to reduce `rbuf` in place. Otherwise `sbuf` and `rbuf` must have the same count and type.
template <
    send_buffer                                            SBuf,
    recv_buffer                                            RBuf,
    mpi::experimental::valid_op<SBuf, RBuf>                Op,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
void allreduce(SBuf&& sbuf, RBuf&& rbuf, Op const& op, Comm const& comm) {
    auto sbuf_ptr = mpi::experimental::ptr(sbuf);
    if (sbuf_ptr == MPI_IN_PLACE) {
        int err = MPI_Allreduce(
            sbuf_ptr,
            mpi::experimental::ptr(rbuf),
            static_cast<int>(mpi::experimental::count(rbuf)),
            mpi::experimental::type(rbuf),
            mpi::experimental::as_mpi_op(op, sbuf, rbuf),
            mpi::experimental::handle(comm)
        );
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    } else {
        using scount_t = decltype(mpi::experimental::count(sbuf));
        KAMPING_V2_ASSERT(
            mpi::experimental::count(sbuf) == static_cast<scount_t>(mpi::experimental::count(rbuf)),
            "send and receive buffer must have the same count"
        );
        KAMPING_V2_ASSERT(
            mpi::experimental::type(sbuf) == mpi::experimental::type(rbuf),
            "send and receive buffer must have the same type"
        );
        int err = MPI_Allreduce(
            sbuf_ptr,
            mpi::experimental::ptr(rbuf),
            static_cast<int>(mpi::experimental::count(sbuf)),
            mpi::experimental::type(sbuf),
            mpi::experimental::as_mpi_op(op, sbuf, rbuf),
            mpi::experimental::handle(comm)
        );
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    }
}
} // namespace mpi::experimental
