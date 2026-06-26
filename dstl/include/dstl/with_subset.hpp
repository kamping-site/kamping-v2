#pragma once

#include <concepts>

#include <kamping/v2/comm.hpp>

#include "dstl/algorithm/all_of.hpp"
#include "mpi/comm.hpp"
#include "mpi/handle.hpp"
namespace dstl {
template <mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm, std::invocable<kamping::v2::comm_view> Func>
void with_subset(Comm const& comm, bool cond, Func&& f) {
    mpi::experimental::comm_view comm_view{mpi::experimental::handle(comm)};
    if (dstl::all_of(cond, comm)) {
        f(comm_view);
        return;
    }
    auto subcomm = comm_view.split(cond);
    if (cond) {
        f(mpi::experimental::comm_view{subcomm});
    }
}
} // namespace dstl
