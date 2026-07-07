// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include "kamping/v2/type_pool.hpp"
#include "mpi/comm.hpp"

namespace kamping::v2 {

/// @brief Non-owning view of both an `MPI_Comm` and a @ref type_pool.
///
/// Borrows its communicator handle and pool by pointer/reference — it neither
/// frees the communicator nor destroys the pool on destruction.  Cheap to copy
/// and pass by value; use it as the standard function-parameter type in generic
/// code that requires MPI datatype registration.
///
/// Satisfies `convertible_to_mpi_handle<MPI_Comm>` via `mpi_handle()`, so it
/// can be passed directly to any `kamping::v2::` collective or point-to-point
/// wrapper.
///
/// Construct from any combination of an existing communicator handle and a
/// pool, or implicitly from a `comm_view` + `type_pool&` pair.  The pointed-to
/// pool must outlive this view.
///
/// @code
/// // Typical class member pattern:
/// class MyGraph {
///     kamping::v2::comm_view  comm_;
///     kamping::v2::type_pool  pool_;
///
///     kamping::v2::comm_view_with_pool env() const { return {comm_, pool_}; }
/// };
///
/// // Generic function accepting a pooled comm:
/// void foo(kamping::v2::comm_view_with_pool comm) {
///     kamping::v2::alltoallv(
///         sbuf | kamping::v2::views::with_auto_pool(comm),
///         kamping::v2::auto_recv_v<MyStruct>(comm),
///         comm);
/// }
/// @endcode
class comm_view_with_pool : public mpi::experimental::comm_accessors<comm_view_with_pool> {
public:
    /// @brief Construct from a raw `MPI_Comm` and an existing pool.
    ///
    /// Both must outlive this view.
    comm_view_with_pool(MPI_Comm comm, type_pool& pool) noexcept : _comm{comm}, _pool{&pool} {}

    /// @brief Construct from a `comm_view` and an existing pool.
    comm_view_with_pool(mpi::experimental::comm_view comm, type_pool& pool) noexcept
        : _comm{comm.mpi_handle()},
          _pool{&pool} {}

    /// @return The underlying `MPI_Comm` (for `handle()` dispatch).
    [[nodiscard]] MPI_Comm mpi_handle() const noexcept {
        return _comm;
    }

    /// @return Reference to the associated type pool.
    [[nodiscard]] type_pool& pool() const noexcept {
        return *_pool;
    }

private:
    MPI_Comm   _comm;
    type_pool* _pool;
};

} // namespace kamping::v2
