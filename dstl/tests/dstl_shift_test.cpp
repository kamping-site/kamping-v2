// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <algorithm>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dstl/algorithm/shift.hpp"
#include "kamping/v2/comm.hpp"

namespace {
// Each rank r holds [r*local_n, r*local_n + 1, ..., r*local_n + local_n - 1].
std::vector<int> iota_local(int rank, int local_n) {
    std::vector<int> v(static_cast<std::size_t>(local_n));
    std::iota(v.begin(), v.end(), rank * local_n);
    return v;
}
} // namespace

// ── shift_left ────────────────────────────────────────────────────────────────

TEST(ShiftLeft, NoOp) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    auto data = iota_local(comm.rank(), 4);
    auto res  = dstl::shift_left(data, 0, comm);
    EXPECT_EQ(res.begin(), data.begin());
    EXPECT_EQ(res.end(), data.end());
    EXPECT_EQ(data, iota_local(comm.rank(), 4));
}

TEST(ShiftLeft, ShiftByTwo) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    int const rank     = comm.rank();
    int const p        = comm.size();
    int const local_n  = 4;
    int const n        = 2;
    auto      data     = iota_local(rank, local_n);

    auto res = dstl::shift_left(data, n, comm);

    // Every rank except the last gets [rank*local_n + n .. (rank+1)*local_n + n).
    // The last rank gets [rank*local_n + n .. rank*local_n + local_n) followed by T{} = 0.
    std::vector<int> expected(static_cast<std::size_t>(local_n));
    for (int i = 0; i < local_n; ++i) {
        int global_src = rank * local_n + i + n;
        expected[static_cast<std::size_t>(i)] = (rank < p - 1 || i < local_n - n) ? global_src : 0;
    }
    EXPECT_EQ(data, expected);

    if (rank < p - 1) {
        EXPECT_EQ(res.begin(), data.begin());
        EXPECT_EQ(res.end(), data.end());
    } else {
        EXPECT_EQ(res.begin(), data.begin());
        EXPECT_EQ(res.end(), data.begin() + (local_n - n));
    }
}

TEST(ShiftLeft, ShiftByLocalSize) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    int const rank    = comm.rank();
    int const p       = comm.size();
    int const local_n = 4;
    auto      data    = iota_local(rank, local_n);

    // n == local_n: each rank's buffer is entirely replaced by the right neighbor's data.
    auto res = dstl::shift_left(data, local_n, comm);

    std::vector<int> expected(static_cast<std::size_t>(local_n), 0);
    if (rank < p - 1) {
        // Receives the full buffer of the right neighbor.
        std::iota(expected.begin(), expected.end(), (rank + 1) * local_n);
        EXPECT_EQ(res.begin(), data.begin());
        EXPECT_EQ(res.end(), data.end());
    } else {
        // Last rank: no right neighbor, entire buffer is T{} = 0.
        EXPECT_EQ(res.begin(), data.begin());
        EXPECT_EQ(res.end(), data.begin());
    }
    EXPECT_EQ(data, expected);
}

// ── shift_right ───────────────────────────────────────────────────────────────

TEST(ShiftRight, NoOp) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    auto data = iota_local(comm.rank(), 4);
    auto res  = dstl::shift_right(data, 0, comm);
    EXPECT_EQ(res.begin(), data.begin());
    EXPECT_EQ(res.end(), data.end());
    EXPECT_EQ(data, iota_local(comm.rank(), 4));
}

TEST(ShiftRight, ShiftByTwo) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    int const rank    = comm.rank();
    int const local_n = 4;
    int const n       = 2;
    auto      data    = iota_local(rank, local_n);

    auto res = dstl::shift_right(data, n, comm);

    // Every rank except rank 0 gets [(rank-1)*local_n + (local_n-n) .. rank*local_n + (local_n-n)).
    // Rank 0 gets T{}=0 in the first n positions, then its original first (local_n-n) elements.
    std::vector<int> expected(static_cast<std::size_t>(local_n));
    for (int i = 0; i < local_n; ++i) {
        int global_dst = rank * local_n + i;
        int global_src = global_dst - n; // element that lands here after the shift
        expected[static_cast<std::size_t>(i)] = (rank > 0 || i >= n) ? global_src : 0;
    }
    EXPECT_EQ(data, expected);

    if (rank > 0) {
        EXPECT_EQ(res.begin(), data.begin());
        EXPECT_EQ(res.end(), data.end());
    } else {
        EXPECT_EQ(res.begin(), data.begin() + n);
        EXPECT_EQ(res.end(), data.end());
    }
}

TEST(ShiftRight, ShiftByLocalSize) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    int const rank    = comm.rank();
    int const local_n = 4;
    auto      data    = iota_local(rank, local_n);

    // n == local_n: each rank's buffer is entirely replaced by the left neighbor's data.
    auto res = dstl::shift_right(data, local_n, comm);

    std::vector<int> expected(static_cast<std::size_t>(local_n), 0);
    if (rank > 0) {
        // Receives the full buffer of the left neighbor.
        std::iota(expected.begin(), expected.end(), (rank - 1) * local_n);
        EXPECT_EQ(res.begin(), data.begin());
        EXPECT_EQ(res.end(), data.end());
    } else {
        // Rank 0: no left neighbor, entire buffer is T{} = 0.
        EXPECT_EQ(res.begin(), data.begin() + local_n);
        EXPECT_EQ(res.end(), data.end());
    }
    EXPECT_EQ(data, expected);
}

// ── single-rank equivalence ────────────────────────────────────────────────────

// With p=1 the distributed versions must match the sequential std algorithms.
TEST(ShiftLeft, SingleRankMatchesStd) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    if (comm.size() != 1) {
        GTEST_SKIP() << "single-rank only";
    }
    std::vector<int> data  = {1, 2, 3, 4, 5, 6};
    std::vector<int> ref   = data;
    int const        n     = 3;
    std::shift_left(ref.begin(), ref.end(), n);
    // Tail gets T{} = 0 in dstl version (MPI_PROC_NULL staging).
    std::fill(ref.end() - n, ref.end(), 0);

    dstl::shift_left(data, n, comm);
    EXPECT_EQ(data, ref);
}

TEST(ShiftRight, SingleRankMatchesStd) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    if (comm.size() != 1) {
        GTEST_SKIP() << "single-rank only";
    }
    std::vector<int> data = {1, 2, 3, 4, 5, 6};
    std::vector<int> ref  = data;
    int const        n    = 2;
    std::shift_right(ref.begin(), ref.end(), n);
    // Head gets T{} = 0 in dstl version (MPI_PROC_NULL staging).
    std::fill(ref.begin(), ref.begin() + n, 0);

    dstl::shift_right(data, n, comm);
    EXPECT_EQ(data, ref);
}
