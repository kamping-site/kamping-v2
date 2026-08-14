// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <algorithm>
#include <compare>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dstl/algorithm/min.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/views/with_type_view.hpp"

namespace {
std::vector<int> iota_vec(int start, int n) {
    std::vector<int> v(static_cast<std::size_t>(n));
    std::iota(v.begin(), v.end(), start);
    return v;
}

// Copy-constructible but not assignable (mirrors std::map's/absl::flat_hash_map's value type,
// std::pair<const K, V>). The generic (non-builtin-op) combine path combines by
// copy-construction, not assignment (see dstl/include/dstl/algorithm/README.md), so this type
// must work end-to-end through an actual reduction, not merely be accepted at the call site.
struct NonAssignable {
    int value;
    NonAssignable(int v) : value(v) {}
    NonAssignable(NonAssignable const&)            = default;
    NonAssignable& operator=(NonAssignable const&) = delete;
    NonAssignable& operator=(NonAssignable&&)      = delete;
    auto           operator<=>(NonAssignable const&) const = default;
};
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// dstl::min
// ─────────────────────────────────────────────────────────────────────────────

// Rank r holds [r*n, r*n+n); the global minimum is on rank 0.
TEST(Min, AscendingBlocks) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    n   = 4;
    auto const                   vec = iota_vec(comm.rank() * n, n);

    EXPECT_EQ(dstl::min(vec, std::ranges::less{}, std::identity{}, comm), 0);
}

// Reversed block order: the global minimum is on the last rank.
TEST(Min, DescendingBlocks) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p   = comm.size();
    int const                    n   = 4;
    auto const                   vec = iota_vec((p - 1 - comm.rank()) * n, n);

    EXPECT_EQ(dstl::min(vec, std::ranges::less{}, std::identity{}, comm), 0);
}

// Positive values on every rank: the minimum is still correctly identified, i.e. no rank's
// contribution is silently clamped or defaulted to 0.
TEST(Min, AllPositive) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    std::vector<int>             data = {10, comm.rank() + 1};
    EXPECT_EQ(dstl::min(data, std::ranges::less{}, std::identity{}, comm), 1);
}

// Even ranks are empty; odd ranks carry the global iota. The empty ranks must contribute the
// sentinel (numeric_limits<int>::max()), never a wrong value.
TEST(Min, EmptyRanksInterleaved) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p    = comm.size();
    int const                    n    = 4;
    int const                    rank = comm.rank();
    std::vector<int>             data;
    if (rank % 2 == 1) {
        data = iota_vec((rank / 2) * n, n);
    }
    int const num_odd = p / 2;
    if (num_odd == 0) {
        GTEST_SKIP() << "needs at least one odd rank";
    }
    EXPECT_EQ(dstl::min(data, std::ranges::less{}, std::identity{}, comm), 0);
}

// An entirely empty global range falls back to the sentinel on every rank, rather than the
// undefined behavior std::ranges::min has on an empty range.
TEST(Min, AllRanksEmpty) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    std::vector<int>             data;
    EXPECT_EQ(dstl::min(data, std::ranges::less{}, std::identity{}, comm), std::numeric_limits<int>::max());
}

// Custom comparator: under std::ranges::greater the "min" is the numerically largest element.
TEST(Min, CustomComparatorGreater) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p   = comm.size();
    int const                    n   = 4;
    auto const                   vec = iota_vec(comm.rank() * n, n);

    EXPECT_EQ(
        dstl::min(vec, std::ranges::greater{}, std::identity{}, comm, std::numeric_limits<int>::min()),
        n * p - 1
    );
}

// Projection: under a negating projection, "min" of the ascending global sequence is its
// largest (most negative-after-negation) element, i.e. the last global element.
TEST(Min, ProjectionNegate) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p   = comm.size();
    int const                    n   = 4;
    auto const                   vec = iota_vec(comm.rank() * n, n);
    auto                         neg = [](int x) {
        return -x;
    };

    EXPECT_EQ(dstl::min(vec, std::ranges::less{}, neg, comm, std::numeric_limits<int>::min()), n * p - 1);
}

// An ordinary, non-const std::pair<int, int> must keep working. std::pair is never
// std::is_trivially_copyable_v (its assignment operators aren't specified as defaulted/trivial,
// even for two ints -- see https://stackoverflow.com/q/58283694), so dstl::min must not gate on
// that trait, only on std::copy_constructible.
TEST(Min, OrdinaryPairValueType) {
    kamping::v2::comm_view const     comm{MPI_COMM_WORLD};
    std::vector<std::pair<int, int>> data   = {{comm.rank(), 0}};
    auto const                       winner = dstl::min(
        data | kamping::v2::views::with_type(MPI_INT),
        std::ranges::less{},
        [](auto const& p) { return p.first; },
        comm,
        std::pair<int, int>{1'000'000, 0}
    );
    EXPECT_EQ(winner.first, 0);
}

// With p == 1 the distributed result must match std::ranges::min.
TEST(Min, SingleRankMatchesStd) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    if (comm.size() != 1) {
        GTEST_SKIP() << "single-rank only";
    }
    std::vector<int> const data = {3, 1, 4, 1, 5, 9, 2, 6};
    EXPECT_EQ(dstl::min(data, std::ranges::less{}, std::identity{}, comm), std::ranges::min(data));
}

// Rank r contributes NonAssignable{r}; the winner must be NonAssignable{0}. Exercises the
// generic (non-builtin-op) combine path end-to-end for a value type that only the
// copy-construct fix (not the old assignment-based one) can handle.
TEST(Min, NonAssignableValueType) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    std::vector<NonAssignable>   data = {NonAssignable{comm.rank()}};

    auto const winner = dstl::min(
        data | kamping::v2::views::with_type(MPI_INT),
        std::ranges::less{},
        std::identity{},
        comm,
        NonAssignable{1'000'000}
    );

    EXPECT_EQ(winner.value, 0);
}
