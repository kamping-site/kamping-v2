// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <span>

#include <mpi.h>
#include <mpi/sentinels.hpp>

namespace kamping::v2 {

using mpi::experimental::bottom;
using mpi::experimental::bottom_t;
using mpi::experimental::inplace;
using mpi::experimental::inplace_t;
using mpi::experimental::null_buf;
using mpi::experimental::null_buf_t;
using mpi::experimental::null_buf_v;
using mpi::experimental::null_buf_v_t;
using mpi::experimental::statuses_ignore;
using mpi::experimental::statuses_ignore_t;

/// Type of the buffer returned by @ref auto_null_recv_v.
//
// Satisfies both `recv_buffer_v` (accepted by core gatherv) and `deferred_recv_buf_v` (so infer() runs the count
// exchange). The deferred-protocol members are no-ops; `mpi_counts()` returns a null span that MPI ignores on
// non-root ranks.
struct auto_null_recv_v_t {
    static void* mpi_ptr() noexcept {
        return nullptr;
    }
    static MPI_Datatype mpi_type() noexcept {
        return MPI_DATATYPE_NULL;
    }
    std::span<int> mpi_counts() noexcept {
        return {};
    }
    std::span<int const> mpi_counts() const noexcept {
        return {};
    }
    static std::span<int const> mpi_displs() noexcept {
        return {};
    }
    static void set_comm_size(int) noexcept {}
    static void commit_counts() noexcept {}
};

/// An empty receive buffer for `gatherv` on non-root ranks, if the root's receive buffer obtains its counts from the
/// others, e.g. `auto_recv_v`: the non-root ranks then still take part in the count exchange.
///
/// @code
/// if (comm.rank() == root) {
///     auto [_, all] = v2::gatherv(local_data, v2::auto_recv_v<int>(), root, comm);
/// } else {
///     v2::gatherv(local_data, v2::auto_null_recv_v(), root, comm);
/// }
/// @endcode
///
/// Pass the result directly, as here; it cannot be used through a `const` variable.
inline auto_null_recv_v_t auto_null_recv_v() noexcept {
    return {};
}

} // namespace kamping::v2
