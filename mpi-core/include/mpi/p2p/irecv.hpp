#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <
    recv_buffer                                RBuf,
    rank                                       Source  = int,
    tag                                        Tag     = int,
    convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void irecv(RBuf&& rbuf, Source source, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Irecv(
        ptr(rbuf),
        static_cast<int>(count(rbuf)),
        type(rbuf),
        to_rank(source),
        to_tag(tag),
        handle(comm),
        handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
