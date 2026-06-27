// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include <dstl/algorithm.hpp>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/comm.hpp"

namespace {
// A contiguous ascending block [start, start + n).
std::vector<int> block(int start, int n) {
    std::vector<int> v(static_cast<std::size_t>(n));
    std::iota(v.begin(), v.end(), start);
    return v;
}

// A contiguous descending block {start, start - 1, ..., start - n + 1}.
std::vector<int> descending_block(int start, int n) {
    std::vector<int> v(static_cast<std::size_t>(n));
    std::iota(v.rbegin(), v.rend(), start - n + 1);
    return v;
}
} // namespace

// ── sorted / unsorted across ranks ──────────────────────────────────────────────

// Rank r holds [r*n, r*n + n): the concatenation is the global iota, hence sorted.
TEST(IsSorted, GloballySortedAscending) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    int const              n    = 4;
    auto                   data = block(comm.rank() * n, n);
    EXPECT_TRUE(dstl::is_sorted(data, comm));
}

// A single rank's slice being locally unsorted makes the whole range unsorted.
// The result is global (all_of), so every rank must observe false.
TEST(IsSorted, LocalViolationOnRankZero) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    int const              n    = 4;
    auto                   data = block(comm.rank() * n, n);
    if (comm.rank() == 0) {
        std::swap(data[0], data[1]); // {1, 0, 2, 3} — locally unsorted
    }
    EXPECT_FALSE(dstl::is_sorted(data, comm));
}

// Each rank's slice is locally sorted, but the block order is reversed across
// ranks so every adjacent rank pair violates the cross-rank boundary.
TEST(IsSorted, CrossRankBoundaryViolation) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    if (comm.size() == 1) {
        GTEST_SKIP() << "needs more than one rank";
    }
    int const n    = 4;
    int const p    = comm.size();
    auto      data = block((p - 1 - comm.rank()) * n, n);
    EXPECT_FALSE(dstl::is_sorted(data, comm));
}

// is_sorted is non-strict: equal elements across the boundary are still sorted.
TEST(IsSorted, EqualElementsAreSorted) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    std::vector<int>       data(4, 7); // identical on every rank
    EXPECT_TRUE(dstl::is_sorted(data, comm));
}

// ── empty ranks ─────────────────────────────────────────────────────────────────

// Odd ranks are empty; the non-empty ranks (indexed by rank/2) carry the global
// iota. with_subset must skip the empty ranks and still report sorted.
TEST(IsSorted, EmptyRanksGloballySorted) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    int const              n    = 4;
    int const              rank = comm.rank();
    std::vector<int>       data;
    if (rank % 2 == 0) {
        data = block((rank / 2) * n, n);
    }
    EXPECT_TRUE(dstl::is_sorted(data, comm));
}

// Same empty layout, but the non-empty blocks are in reversed order, so the
// boundary between the (non-adjacent) non-empty ranks is violated.
TEST(IsSorted, EmptyRanksCrossViolation) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    int const              p           = comm.size();
    int const              num_nonempty = (p + 1) / 2;
    if (num_nonempty < 2) {
        GTEST_SKIP() << "needs at least two non-empty ranks";
    }
    int const        n    = 4;
    int const        rank = comm.rank();
    std::vector<int> data;
    if (rank % 2 == 0) {
        int const idx = rank / 2;
        data          = block((num_nonempty - 1 - idx) * n, n);
    }
    EXPECT_FALSE(dstl::is_sorted(data, comm));
}

// An entirely empty global range is sorted.
TEST(IsSorted, AllRanksEmpty) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    std::vector<int>       data;
    EXPECT_TRUE(dstl::is_sorted(data, comm));
}

// ── custom comparator / projection ──────────────────────────────────────────────

// Global descending sequence; sorted under std::ranges::greater, unsorted under
// the default ascending comparator.
TEST(IsSorted, CustomComparatorDescending) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    int const              n    = 4;
    int const              p    = comm.size();
    int const              rank = comm.rank();
    // Global descending list p*n-1 .. 0; rank r holds positions [r*n, r*n+n).
    auto data = descending_block(p * n - 1 - rank * n, n);
    EXPECT_TRUE(dstl::is_sorted(data, comm, std::ranges::greater{}));
    if (p * n > 1) {
        EXPECT_FALSE(dstl::is_sorted(data, comm));
    }
}

// The same descending data is ascending under a negating projection.
TEST(IsSorted, ProjectionNegate) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    int const              n    = 4;
    int const              p    = comm.size();
    int const              rank = comm.rank();
    auto                   data = descending_block(p * n - 1 - rank * n, n);
    auto                   neg  = [](int x) { return -x; };
    EXPECT_TRUE(dstl::is_sorted(data, comm, std::ranges::less{}, neg));
}

// ── single-rank equivalence ─────────────────────────────────────────────────────

// With p == 1 the distributed result must match std::ranges::is_sorted.
TEST(IsSorted, SingleRankMatchesStd) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    if (comm.size() != 1) {
        GTEST_SKIP() << "single-rank only";
    }
    std::vector<int> const sorted_data   = {1, 2, 2, 3, 5};
    std::vector<int> const unsorted_data = {1, 3, 2};
    EXPECT_EQ(dstl::is_sorted(sorted_data, comm), std::ranges::is_sorted(sorted_data));
    EXPECT_EQ(dstl::is_sorted(unsorted_data, comm), std::ranges::is_sorted(unsorted_data));
    EXPECT_TRUE(dstl::is_sorted(sorted_data, comm));
    EXPECT_FALSE(dstl::is_sorted(unsorted_data, comm));
}
