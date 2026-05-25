// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <climits>

#include <mpi.h>

#include "kamping/kassert/kassert.hpp"
#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <
    recv_buffer                                RBuf,
    convertible_to_mpi_handle_ptr<MPI_Message> Message,
    convertible_to_mpi_handle_ptr<MPI_Request> Request>
void imrecv(RBuf&& rbuf, Message&& message, Request&& request) {
#if MPI_VERSION >= 4
    int err = MPI_Imrecv_c(ptr(rbuf), count(rbuf), type(rbuf), handle_ptr(message), handle_ptr(request));
#else
    KAMPING_ASSERT(count(rbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    int err =
        MPI_Imrecv(ptr(rbuf), static_cast<int>(count(rbuf)), type(rbuf), handle_ptr(message), handle_ptr(request));
#endif
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
