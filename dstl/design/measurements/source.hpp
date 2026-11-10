// Design sketch — NOT a build target. Illustrates the `source` concept and the
// wall_clock / manual_clock / value_counter meters.
//
// A source is the second customization point (mirroring buffer_traits /
// handle_traits): it says how to sample a meter and how to turn two samples into
// a measured value. `Timer` and `Counter` are then just `tree<Source>` with a
// different source — not separate classes. `manual_clock` makes the entire
// pipeline deterministically unit-testable.

#pragma once

#include <concepts>
#include <cstdint>

#include <mpi.h>

namespace kamping::v2::measure::source {

/// @brief A meter. `sample()` reads the current value; `delta(before, after)`
/// turns two samples into a measured `value_type`.
template <class S>
concept source_concept = requires(S s, typename S::sample_type a, typename S::sample_type b) {
    typename S::sample_type; // e.g. a time point or a raw counter reading
    typename S::value_type;  // what a measurement yields (seconds, bytes, cycles …)
    { s.sample() } -> std::same_as<typename S::sample_type>;
    { s.delta(a, b) } -> std::same_as<typename S::value_type>; // after - before
};

/// @brief Wall-clock time via MPI_Wtime, in seconds. The default timer source.
struct wall_clock {
    using sample_type = double;
    using value_type  = double;
    sample_type sample() const { return MPI_Wtime(); }
    value_type  delta(sample_type before, sample_type after) const { return after - before; }
};
static_assert(source_concept<wall_clock>);

/// @brief A settable clock for deterministic tests. Advance it by hand; the
/// whole record→reduce→present pipeline becomes reproducible without real time.
struct manual_clock {
    using sample_type = double;
    using value_type  = double;
    double now = 0.0;
    void   advance(double dt) { now += dt; }
    sample_type sample() const { return now; }
    value_type  delta(sample_type before, sample_type after) const { return after - before; }
};
static_assert(source_concept<manual_clock>);

/// @brief A counter meter: each measurement contributes a user-supplied integer
/// increment rather than a time delta. `sample()` is unused (the value is passed
/// explicitly at record time); modelled here as a no-op reading for uniformity.
template <typename T = std::int64_t>
struct value_counter {
    using sample_type = T;
    using value_type  = T;
    sample_type sample() const { return T{}; }
    value_type  delta(sample_type before, sample_type after) const { return after - before; }
};
static_assert(source_concept<value_counter<>>);

} // namespace kamping::v2::measure::source
