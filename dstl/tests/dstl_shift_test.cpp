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
    if (p == 1) {
        // n == total: no effects (matches std::shift_left semantics).
        expected = iota_local(rank, local_n);
        EXPECT_EQ(res.begin(), data.begin());
        EXPECT_EQ(res.end(), data.end());
    } else if (rank < p - 1) {
        // Receives the full buffer of the right neighbor.
        std::iota(expected.begin(), expected.end(), (rank + 1) * local_n);
        EXPECT_EQ(res.begin(), data.begin());
        EXPECT_EQ(res.end(), data.end());
    } else {
        // Last rank (p>1): no right neighbor, entire buffer is T{} = 0.
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
    int const p       = comm.size();
    int const local_n = 4;
    auto      data    = iota_local(rank, local_n);

    // n == local_n: each rank's buffer is entirely replaced by the left neighbor's data.
    auto res = dstl::shift_right(data, local_n, comm);

    std::vector<int> expected(static_cast<std::size_t>(local_n), 0);
    if (p == 1) {
        // n == total: no effects (matches std::shift_right semantics).
        expected = iota_local(rank, local_n);
        EXPECT_EQ(res.begin(), data.begin());
        EXPECT_EQ(res.end(), data.end());
    } else if (rank > 0) {
        // Receives the full buffer of the left neighbor.
        std::iota(expected.begin(), expected.end(), (rank - 1) * local_n);
        EXPECT_EQ(res.begin(), data.begin());
        EXPECT_EQ(res.end(), data.end());
    } else {
        // Rank 0 (p>1): no left neighbor, entire buffer is T{} = 0.
        EXPECT_EQ(res.begin(), data.begin() + local_n);
        EXPECT_EQ(res.end(), data.end());
    }
    EXPECT_EQ(data, expected);
}

// ── irregular input (n > local size on some ranks) ────────────────────────────

// shift_left by more than local_size of rank 0 (rank 0 has 1 element, n=3).
// Requires 4 ranks.  Rank distributions: [1, 2, 3, 2] → sizes vary.
// Global sequence: 0 | 1 2 | 3 4 5 | 6 7
// After shift_left(3): 3 | 4 5 6 | 7 0 0 | 0 0
TEST(ShiftLeft, IrregularNExceedsLocalSizeOnSomeRanks) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    if (comm.size() != 4) {
        GTEST_SKIP() << "requires exactly 4 ranks";
    }
    int const rank = comm.rank();

    // local_sizes: rank 0 → 1, rank 1 → 2, rank 2 → 3, rank 3 → 2
    std::vector<int> const local_sizes = {1, 2, 3, 2};
    int const              local_n     = local_sizes[static_cast<std::size_t>(rank)];

    // offsets: rank 0 → 0, 1 → 1, 2 → 3, 3 → 6
    std::vector<int> const offsets = {0, 1, 3, 6};

    std::vector<int> data(static_cast<std::size_t>(local_n));
    std::iota(data.begin(), data.end(), offsets[static_cast<std::size_t>(rank)]);

    int const n = 3; // exceeds local size of rank 0 (=1) and rank 1 (=2)

    auto res = dstl::shift_left(data, n, comm);

    // Expected output: global index i receives global element i+3 (or 0 if i+3 >= 8).
    std::vector<int> expected(static_cast<std::size_t>(local_n), 0);
    int const        total   = 8;
    int const        my_off  = offsets[static_cast<std::size_t>(rank)];
    int              valid_n = 0;
    for (int i = 0; i < local_n; ++i) {
        int global_dst = my_off + i;
        int global_src = global_dst + n;
        if (global_src < total) {
            expected[static_cast<std::size_t>(i)] = global_src;
            ++valid_n;
        }
    }
    // Elements outside [res.begin(), res.end()) are in a valid but unspecified state.
    EXPECT_EQ(res.begin(), data.begin());
    EXPECT_EQ(res.end(), data.begin() + valid_n);
    EXPECT_EQ(
        std::vector<int>(res.begin(), res.end()),
        std::vector<int>(expected.begin(), expected.begin() + valid_n));
}

// shift_right by more than local_size of the last rank (rank 3 has 2 elements, n=3).
// Uses the same irregular distribution as above.
// Global sequence: 0 | 1 2 | 3 4 5 | 6 7
// After shift_right(3): 0 0 0 | 0 1 2 | 3 4 5 | 6 7  — wait, total=8, n=3
// Correct: 0 | 1 2 | 3 4 5 | 6 7  → shift right 3 → pos i gets element i-3
// pos 0..2 → 0 (no source), pos 3 → 0, pos 4 → 1, pos 5 → 2, pos 6 → 3, pos 7 → 4
// rank 0 (off=0, size=1): pos 0 → 0        → [0]
// rank 1 (off=1, size=2): pos 1 → 0, 2 → 0 → [0, 0]  -- wait
// Let me recalculate: element at global pos i+n comes from global element i.
// After shift_right(3): global pos j gets element j-3 (or 0 if j<3).
// rank 0 (off=0, size=1): j=0 → 0           → [0]
// rank 1 (off=1, size=2): j=1→0, j=2→0      → [0, 0]   ← wait j=1 → j-3 = -2 < 0 → 0
// Actually: j=1 → src=j-3=-2 → 0; j=2 → src=-1 → 0
// rank 2 (off=3, size=3): j=3→0, j=4→1, j=5→2 → [0, 1, 2]
// rank 3 (off=6, size=2): j=6→3, j=7→4          → [3, 4]
// rank 0 valid_recv_start = min(1, max(0, 3-0))=1 → res = {end, end}
// rank 1 valid_recv_start = min(2, max(0, 3-1))=2 → res = {end, end}
// rank 2 valid_recv_start = min(3, max(0, 3-3))=0 → res = {begin, end}
// rank 3 valid_recv_start = 0 → res = {begin, end}
TEST(ShiftRight, IrregularNExceedsLocalSizeOnSomeRanks) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    if (comm.size() != 4) {
        GTEST_SKIP() << "requires exactly 4 ranks";
    }
    int const rank = comm.rank();

    std::vector<int> const local_sizes = {1, 2, 3, 2};
    int const              local_n     = local_sizes[static_cast<std::size_t>(rank)];
    std::vector<int> const offsets     = {0, 1, 3, 6};

    std::vector<int> data(static_cast<std::size_t>(local_n));
    std::iota(data.begin(), data.end(), offsets[static_cast<std::size_t>(rank)]);

    int const n = 3;

    auto res = dstl::shift_right(data, n, comm);

    // Global pos j gets element j-n (or 0 if j < n).
    std::vector<int> expected(static_cast<std::size_t>(local_n), 0);
    int const        my_off           = offsets[static_cast<std::size_t>(rank)];
    int              valid_recv_start = 0;
    for (int i = 0; i < local_n; ++i) {
        int global_dst = my_off + i;
        int global_src = global_dst - n;
        if (global_src >= 0) {
            expected[static_cast<std::size_t>(i)] = global_src;
        } else {
            ++valid_recv_start;
        }
    }
    // Elements outside [res.begin(), res.end()) are in a valid but unspecified state.
    EXPECT_EQ(res.begin(), data.begin() + valid_recv_start);
    EXPECT_EQ(res.end(), data.end());
    EXPECT_EQ(
        std::vector<int>(res.begin(), res.end()),
        std::vector<int>(expected.begin() + valid_recv_start, expected.end()));
}

// shift_left where n >= total global size: no effects (matches std::shift_left).
TEST(ShiftLeft, ShiftByTotalOrMoreIsNoOp) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    int const rank    = comm.rank();
    int const p       = comm.size();
    int const local_n = 3;
    auto const original = iota_local(rank, local_n);
    auto       data     = original;
    int const  total    = p * local_n;

    auto res = dstl::shift_left(data, total, comm);

    EXPECT_EQ(data, original);
    EXPECT_EQ(res.begin(), data.begin());
    EXPECT_EQ(res.end(), data.end());
}

// shift_right where n >= total global size: no effects (matches std::shift_right).
TEST(ShiftRight, ShiftByTotalOrMoreIsNoOp) {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    int const rank    = comm.rank();
    int const p       = comm.size();
    int const local_n = 3;
    auto const original = iota_local(rank, local_n);
    auto       data     = original;
    int const  total    = p * local_n;

    auto res = dstl::shift_right(data, total, comm);

    EXPECT_EQ(data, original);
    EXPECT_EQ(res.begin(), data.begin());
    EXPECT_EQ(res.end(), data.end());
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
