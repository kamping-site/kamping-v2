// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dstl/algorithm/max.hpp"
#include "kamping/v2/comm.hpp"

namespace {
std::vector<int> iota_vec(int start, int n) {
    std::vector<int> v(static_cast<std::size_t>(n));
    std::iota(v.begin(), v.end(), start);
    return v;
}
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// dstl::max
// ─────────────────────────────────────────────────────────────────────────────

// Rank r holds [r*n, r*n+n); the global maximum is on the last rank.
TEST(Max, AscendingBlocks) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p   = comm.size();
    int const                    n   = 4;
    auto const                   vec = iota_vec(comm.rank() * n, n);

    EXPECT_EQ(dstl::max(vec, std::ranges::less{}, std::identity{}, comm), n * p - 1);
}

// Reversed block order: the global maximum is on rank 0.
TEST(Max, DescendingBlocks) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p   = comm.size();
    int const                    n   = 4;
    auto const                   vec = iota_vec((p - 1 - comm.rank()) * n, n);

    EXPECT_EQ(dstl::max(vec, std::ranges::less{}, std::identity{}, comm), n * p - 1);
}

// Negative values on every rank: the maximum is still correctly identified, i.e. no rank's
// contribution is silently clamped or defaulted to 0.
TEST(Max, AllNegative) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    std::vector<int>             data = {-10, -(comm.rank() + 1)};
    EXPECT_EQ(dstl::max(data, std::ranges::less{}, std::identity{}, comm), -1);
}

// Even ranks are empty; odd ranks carry the global iota. The empty ranks must contribute the
// sentinel (numeric_limits<int>::lowest()), never a wrong value.
TEST(Max, EmptyRanksInterleaved) {
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
    EXPECT_EQ(dstl::max(data, std::ranges::less{}, std::identity{}, comm), num_odd * n - 1);
}

// An entirely empty global range falls back to the sentinel on every rank, rather than the
// undefined behavior std::ranges::max has on an empty range.
TEST(Max, AllRanksEmpty) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    std::vector<int>             data;
    EXPECT_EQ(dstl::max(data, std::ranges::less{}, std::identity{}, comm), std::numeric_limits<int>::lowest());
}

// Custom comparator: under std::ranges::greater the "max" is the numerically smallest element.
TEST(Max, CustomComparatorGreater) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p   = comm.size();
    int const                    n   = 4;
    auto const                   vec = iota_vec(comm.rank() * n, n);

    EXPECT_EQ(dstl::max(vec, std::ranges::greater{}, std::identity{}, comm, std::numeric_limits<int>::max()), 0);
    (void)p;
}

// Projection: under a negating projection, "max" of the ascending global sequence is its
// smallest (most negative-after-negation) element, i.e. global element 0.
TEST(Max, ProjectionNegate) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    n   = 4;
    auto const                   vec = iota_vec(comm.rank() * n, n);
    auto                         neg = [](int x) {
        return -x;
    };

    EXPECT_EQ(dstl::max(vec, std::ranges::less{}, neg, comm, std::numeric_limits<int>::max()), 0);
}

// With p == 1 the distributed result must match std::ranges::max.
TEST(Max, SingleRankMatchesStd) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    if (comm.size() != 1) {
        GTEST_SKIP() << "single-rank only";
    }
    std::vector<int> const data = {3, 1, 4, 1, 5, 9, 2, 6};
    EXPECT_EQ(dstl::max(data, std::ranges::less{}, std::identity{}, comm), std::ranges::max(data));
}
