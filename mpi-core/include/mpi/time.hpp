// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <chrono>

#include <mpi.h>

namespace mpi::experimental {

/// @brief A std::chrono compatible clock type build around `MPI_Wtime`.
struct clock {
    using rep        = double;        //< arithmetic type representing the tick period of the clock, in seconds
    using period     = std::ratio<1>; //< a <std::ratio> type representing the tick period of the clock, in seconds
    using duration   = std::chrono::duration<rep, period>;       //< the duration type
    using time_point = std::chrono::time_point<clock, duration>; //< the time_point type

    // MPI_Wtime returns elapsed wall-clock time since a fixed
    // point in the past that does not change during the process lifetime.
    // We interpret this as a monotonic (steady) clock.
    static constexpr bool is_steady = false; //< steady clock flag, always true

    /// @return A time point representing the current time, as returned by `MPI_Wtime`.
    static time_point now() noexcept {
        return time_point{duration{MPI_Wtime()}};
    }

    /// @return A duration represent the clocks resolution, i.e. the number of seconds between clock ticks as returned
    /// by `MPI_Wtick`.
    static duration resolution() noexcept {
        return clock::duration{MPI_Wtick()};
    }
};

// FIXME: not supported by AppleClang 21.0.0.21000101
// static_assert(std::chrono::is_clock_v<clock>);

} // namespace mpi::experimental
