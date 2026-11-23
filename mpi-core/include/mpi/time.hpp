// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <chrono>

#include <mpi.h>

/// @file
/// `clock`, a `std::chrono` clock reading `MPI_Wtime`.

namespace mpi::experimental {

/// @brief A `std::chrono` clock built around `MPI_Wtime`.
///
/// Its durations are `std::chrono::duration<double>` in seconds and can be sent and reduced like any builtin type,
/// e.g. to find the slowest process:
///
/// @code
/// auto const start   = clock::now();
/// work();
/// auto       elapsed = clock::now() - start;
/// allreduce(inplace, views::ref_single(elapsed), ops::max<>{}, comm);
/// @endcode
///
/// Time points of different processes are comparable only if the MPI implementation synchronizes the clocks
/// (`MPI_WTIME_IS_GLOBAL`); durations are always comparable.
struct clock {
    using rep        = double;                                   ///< The number of seconds.
    using period     = std::ratio<1>;                            ///< The tick period: one second.
    using duration   = std::chrono::duration<rep, period>;       ///< The duration type, in seconds.
    using time_point = std::chrono::time_point<clock, duration>; ///< The time point type.

    /// The clock is steady: MPI specifies `MPI_Wtime` as the time since some point in the past that does not change
    /// during the life of the process.
    static constexpr bool is_steady = true;

    /// @return The current time, as returned by `MPI_Wtime`.
    static time_point now() noexcept {
        return time_point{duration{MPI_Wtime()}};
    }

    /// @return The resolution of the clock, the time between two ticks, as returned by `MPI_Wtick`.
    static duration resolution() noexcept {
        return duration{MPI_Wtick()};
    }
};

// FIXME: not supported by AppleClang 21.0.0.21000101
// static_assert(std::chrono::is_clock_v<clock>);

} // namespace mpi::experimental
