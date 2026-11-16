// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

/// @file
/// @brief MPI thread support levels.

#pragma once

#include <compare>

#include <mpi.h>

namespace mpi::experimental {

/// @brief MPI thread support levels defining the allowed concurrency of MPI calls relative to application threads.
///
/// The levels are ordered: a higher level allows everything a lower one does. The underlying value is the MPI
/// constant, which you can obtain by casting to \c int.
enum class thread_level : int {
    /// No thread support; only one thread may execute, and only the main thread can make MPI calls.
    single = MPI_THREAD_SINGLE,
    /// Only the main thread will make MPI calls, but the application may be multi-threaded.
    funneled = MPI_THREAD_FUNNELED,
    /// Multiple threads may exist, but only one at a time will make MPI calls.
    serialized = MPI_THREAD_SERIALIZED,
    /// Full thread support; multiple threads may call MPI concurrently.
    multiple = MPI_THREAD_MULTIPLE
};

/// @brief Three-way comparison for \ref thread_level by underlying integer value.
inline auto operator<=>(thread_level lhs, thread_level rhs) noexcept {
    return static_cast<int>(lhs) <=> static_cast<int>(rhs);
}

} // namespace mpi::experimental
