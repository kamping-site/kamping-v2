// Design sketch — NOT a build target. Illustrates the `aggregator` concept and
// the `stats` monoid as they would read against `kamping::v2::make_op`.
//
// The whole measurements design rests on this one abstraction: an aggregator is
// a commutative monoid (identity `A{}`, associative+commutative `merge`), which
// is exactly what `make_op` lifts into an `MPI_Op`. Local aggregation is `add`;
// cross-rank aggregation is `merge`; a missing region on a rank is the identity.

#pragma once

#include <cmath>
#include <cstdint>

#include "kamping/v2/ops.hpp" // kamping::v2::make_op, commutative

namespace kamping::v2::measure {

/// @brief A mergeable aggregator: a commutative monoid over samples of type
/// `input`. `A{}` is the identity ("empty"/absent), `add` folds one local
/// sample, `merge` combines two partial results (associative + commutative),
/// and `value()` reads out the final result (which may differ from the internal
/// representation, e.g. Welford's `m2` vs. the reported stddev).
template <class A>
concept aggregator = requires(A a, A const b, typename A::input v) {
    A{};                 // identity element
    a.add(v);            // fold one sample
    a.merge(b);          // combine two partials — must be associative + commutative
    a.value();           // final read-out
};

/// @brief Welford accumulator: mergeable mean / stddev / min / max, plus the
/// rank that held the maximum (straggler identification). Trivially copyable, so
/// `std::vector<stats>` gets an MPI datatype for free via the kamping::types
/// `trivially_copyable` byte-blob adapter, and `make_op<stats>` interprets it.
struct stats {
    using input = double;

    std::uint64_t n           = 0;
    double        mean        = 0.0;
    double        m2          = 0.0;          ///< sum of squared deviations
    double        min         = +INFINITY;
    double        max         = -INFINITY;
    int           argmax_rank = -1;           ///< rank owning `max` (filled at merge time)

    /// @brief Fold one local sample (streaming Welford update).
    void add(double x) {
        ++n;
        double const d = x - mean;
        mean += d / static_cast<double>(n);
        m2 += d * (x - mean);
        min = x < min ? x : min;
        max = x > max ? x : max;
    }

    /// @brief Combine two partials — associative + commutative (chunked Welford).
    /// `other.argmax_rank` is assumed already set to its origin rank.
    void merge(stats const& o) {
        if (o.n == 0) {
            return;                            // identity: merge with empty is a no-op
        }
        if (n == 0) {
            *this = o;
            return;
        }
        double const na = static_cast<double>(n);
        double const nb = static_cast<double>(o.n);
        double const nt = na + nb;
        double const d  = o.mean - mean;
        mean  = mean + d * (nb / nt);
        m2    = m2 + o.m2 + d * d * (na * nb / nt);
        n     = n + o.n;
        min   = o.min < min ? o.min : min;
        if (o.max > max) {
            max         = o.max;
            argmax_rank = o.argmax_rank;
        }
    }

    /// @brief Final read-out.
    struct result {
        std::uint64_t n;
        double        mean, stddev, min, max;
        int           argmax_rank;
    };
    result value() const {
        return {n, mean, n ? std::sqrt(m2 / static_cast<double>(n)) : 0.0, min, max, argmax_rank};
    }
};

static_assert(aggregator<stats>);

/// @brief Rides along in the same reduce to report how many ranks contributed a
/// given region (honest scope reporting for divergent trees).
struct present_count {
    using input = double;                     // value ignored; presence is what matters
    std::uint64_t ranks = 0;
    void          add(double) { ranks = 1; }  // per rank: "I was here"
    void          merge(present_count const& o) { ranks += o.ranks; }
    std::uint64_t value() const { return ranks; }
};
static_assert(aggregator<present_count>);

/// @brief The cross-rank reduction operator for any POD aggregator `A`:
/// `merge` lifted into an `MPI_Op`. This is the single line that connects the
/// measurements suite to the existing ops layer.
///
/// A partial merge is commutative, so we opt into `commutative` for the O(log p)
/// tree reduction. A whole `std::vector<A>` (one slot per canonical region path,
/// identity where absent) is reduced elementwise with this op.
template <aggregator A>
inline auto merge_op() {
    return kamping::v2::make_op<A>([](A const& x, A const& y) {
        A r = x;
        r.merge(y);
        return r;
    }, kamping::v2::commutative);
}

} // namespace kamping::v2::measure
