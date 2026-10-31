// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

/// @file
/// @brief MPI thread support levels.

#pragma once

#include <compare>

#include <mpi.h>

namespace mpi::experimental {

/// @brief MPI thread support levels defining the allowed concurrency of MPI calls relative to application threads.
/// You can obtain the underlying value by casting to \c int.
enum class ThreadLevel : int {
    /// No thread support; only one thread may execute, and only the main thread can make MPI calls.
    single = MPI_THREAD_SINGLE,
    /// Only the main thread will make MPI calls, but the application may be multi-threaded.
    funneled = MPI_THREAD_FUNNELED,
    /// Multiple threads may exist, but only one at a time will make MPI calls.
    serialized = MPI_THREAD_SERIALIZED,
    /// Full thread support; multiple threads may call MPI concurrently.
    multiple = MPI_THREAD_MULTIPLE
};

/// @brief Three-way comparison for \ref ThreadLevel by underlying integer value.
inline auto operator<=>(ThreadLevel lhs, ThreadLevel rhs) noexcept {
    return static_cast<int>(lhs) <=> static_cast<int>(rhs);
}

} // namespace mpi::experimental
