#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <
    send_buffer                         SBuf,
    rank                                Dest = int,
    tag                                 Tag  = int,
    convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void send(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Send(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        to_rank(dest),
        to_tag(tag),
        handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    send_buffer                         SBuf,
    rank                                Dest = int,
    tag                                 Tag  = int,
    convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void bsend(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Bsend(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        to_rank(dest),
        to_tag(tag),
        handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    send_buffer                         SBuf,
    rank                                Dest = int,
    tag                                 Tag  = int,
    convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void ssend(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Ssend(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        to_rank(dest),
        to_tag(tag),
        handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    send_buffer                         SBuf,
    rank                                Dest = int,
    tag                                 Tag  = int,
    convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void rsend(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Rsend(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        to_rank(dest),
        to_tag(tag),
        handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
