// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>

#include <kamping/v2/comm.hpp>

#include "dstl/algorithm/all_of.hpp"
#include "kamping/v2/handle.hpp"
namespace dstl {
/// Calls `f` with a communicator of the processes for which `cond` is true (collective).
///
/// If `cond` holds on all processes, `f` is called with `comm` itself. Otherwise a subcommunicator of the processes
/// with `cond == true` is split off, and only those call `f` with it.
template <kamping::v2::convertible_to_mpi_handle<MPI_Comm> Comm, std::invocable<kamping::v2::comm_view> Func>
void with_subset(Comm const& comm, bool cond, Func&& f) {
    kamping::v2::comm_view comm_view{kamping::v2::handle(comm)};
    if (dstl::all_of(cond, comm)) {
        f(comm_view);
        return;
    }
    auto subcomm = comm_view.split(cond);
    if (cond) {
        f(kamping::v2::comm_view{subcomm});
    }
}
} // namespace dstl
