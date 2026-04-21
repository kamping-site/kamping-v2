#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

template <
    send_buffer                                SBuf,
    recv_buffer                                RBuf,
    rank                                       Dest    = int,
    rank                                       Source  = int,
    tag                                        SendTag = int,
    tag                                        RecvTag = int,
    convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void isendrecv(
    SBuf&&      sbuf,
    Dest        dest,
    SendTag     send_tag,
    RBuf&&      rbuf,
    Source      source,
    RecvTag     recv_tag,
    Comm const& comm,
    Request&&   request
) {
    int err = MPI_Isendrecv(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        to_rank(dest),
        to_tag(send_tag),
        ptr(rbuf),
        static_cast<int>(count(rbuf)),
        type(rbuf),
        to_rank(source),
        to_tag(recv_tag),
        handle(comm),
        handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

} // namespace mpi::experimental
