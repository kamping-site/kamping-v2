#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <
    recv_buffer                                RBuf,
    convertible_to_mpi_handle_ptr<MPI_Message> Message,
    convertible_to_mpi_handle_ptr<MPI_Request> Request>
void imrecv(RBuf&& rbuf, Message&& message, Request&& request) {
    int err = MPI_Imrecv(
        ptr(rbuf),
        static_cast<int>(count(rbuf)),
        type(rbuf),
        handle_ptr(message),
        handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
