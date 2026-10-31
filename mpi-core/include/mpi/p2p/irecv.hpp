// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
/// Starts receiving a message from `source` with `tag` into `rbuf` (`MPI_Irecv`). The caller must keep `rbuf` alive and
/// complete `request`.
template <
    recv_buffer                                RBuf,
    rank                                       Source,
    tag                                        Tag,
    convertible_to_mpi_handle<MPI_Comm>        Comm,
    convertible_to_mpi_handle_ptr<MPI_Request> Request>
void irecv(RBuf&& rbuf, Source source, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Irecv(
        mpi::experimental::ptr(rbuf),
        static_cast<int>(mpi::experimental::count(rbuf)),
        mpi::experimental::type(rbuf),
        mpi::experimental::to_rank(source),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        mpi::experimental::handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
