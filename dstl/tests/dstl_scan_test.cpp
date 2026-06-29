// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <list>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dstl/algorithm/exclusive_scan.hpp"
#include "dstl/algorithm/inclusive_scan.hpp"
#include "dstl/algorithm/reduce.hpp"
#include "kamping/v2/comm.hpp"

// ── helpers ───────────────────────────────────────────────────────────────────

namespace {
std::vector<int> iota_vec(int start, int n) {
    std::vector<int> v(static_cast<std::size_t>(n));
    std::iota(v.begin(), v.end(), start);
    return v;
}
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// dstl::reduce
// ─────────────────────────────────────────────────────────────────────────────

TEST(Reduce, Sum) {
    // Each rank holds [rank*4, rank*4+1, rank*4+2, rank*4+3].
    // Global sum = 0+1+...+(4p-1) = 4p*(4p-1)/2.
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p   = comm.size();
    int const                    n   = 4;
    auto const                   vec = iota_vec(comm.rank() * n, n);

    int const result   = dstl::reduce(vec, 0, std::plus<int>{}, comm);
    int const expected = n * p * (n * p - 1) / 2;
    EXPECT_EQ(result, expected);
}

TEST(Reduce, WithInit) {
    // init is added to the global sum.
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    n   = 2;
    auto const                   vec = iota_vec(comm.rank() * n, n);

    int const total    = n * comm.size() * (n * comm.size() - 1) / 2;
    int const result   = dstl::reduce(vec, 100, std::plus<int>{}, comm);
    EXPECT_EQ(result, 100 + total);
}

TEST(Reduce, EmptyLocal) {
    // All ranks have empty ranges — result should equal init.
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    std::vector<int> const       empty;

    int const result = dstl::reduce(empty, 42, std::plus<int>{}, comm);
    EXPECT_EQ(result, 42);
}

TEST(Reduce, ForwardRangeInput) {
    // Smoke test: input is a std::list (forward_range, not contiguous).
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    std::list<int> const         vals = {1, 2, 3};

    int const result = dstl::reduce(vals, 0, std::plus<int>{}, comm);
    EXPECT_EQ(result, 6 * comm.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// dstl::exclusive_scan
// ─────────────────────────────────────────────────────────────────────────────

TEST(ExclusiveScan, UniformChunks) {
    // Each rank holds [rank*4, ..., rank*4+3].
    // Global exclusive scan: out[i] = 0+1+...+(global_i - 1).
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    rank = comm.rank();
    int const                    n    = 4;
    int const                    base = rank * n;
    auto const                   in   = iota_vec(base, n);
    std::vector<int>             out(static_cast<std::size_t>(n), 0);

    dstl::exclusive_scan(in, out, 0, std::plus<int>{}, comm);

    // Global element i has value 0+1+...+(i-1) = i*(i-1)/2.
    for (int i = 0; i < n; ++i) {
        int const global_i = base + i;
        EXPECT_EQ(out[static_cast<std::size_t>(i)], global_i * (global_i - 1) / 2) << "rank=" << rank << " i=" << i;
    }
}

TEST(ExclusiveScan, SingleElement) {
    // Each rank holds a single element equal to rank.
    // Exclusive scan: out[i] = 0+1+...+(i-1) = i*(i-1)/2.
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    rank = comm.rank();
    std::vector<int>             in   = {rank};
    std::vector<int>             out(1);

    dstl::exclusive_scan(in, out, 0, std::plus<int>{}, comm);

    EXPECT_EQ(out[0], rank * (rank - 1) / 2);
}

TEST(ExclusiveScan, WithInit) {
    // Non-zero init shifts the entire output.
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    rank = comm.rank();
    std::vector<int>             in   = {1};
    std::vector<int>             out(1);

    dstl::exclusive_scan(in, out, 10, std::plus<int>{}, comm);

    // out[0] = init + sum of elements on ranks 0..rank-1 = 10 + rank.
    EXPECT_EQ(out[0], 10 + rank);
}

TEST(ExclusiveScan, ForwardRangeInput) {
    // Smoke test: std::list is a forward_range but not contiguous.
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    std::list<int>               in = {1, 2, 3};
    std::vector<int>             out(3);

    // Does not throw; correctness identical to vector case.
    dstl::exclusive_scan(in, out, 0, std::plus<int>{}, comm);

    int const prefix = 6 * comm.rank(); // each rank contributes 1+2+3 = 6
    EXPECT_EQ(out[0], prefix + 0);
    EXPECT_EQ(out[1], prefix + 1);
    EXPECT_EQ(out[2], prefix + 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// dstl::inclusive_scan
// ─────────────────────────────────────────────────────────────────────────────

TEST(InclusiveScan, UniformChunks) {
    // Each rank holds [rank*4, ..., rank*4+3].
    // Global inclusive scan: out[i] = 0+1+...+global_i.
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    rank = comm.rank();
    int const                    n    = 4;
    int const                    base = rank * n;
    auto const                   in   = iota_vec(base, n);
    std::vector<int>             out(static_cast<std::size_t>(n), 0);

    dstl::inclusive_scan(in, out, std::plus<int>{}, comm);

    // Global element i has value 0+1+...+i = i*(i+1)/2.
    for (int i = 0; i < n; ++i) {
        int const global_i = base + i;
        EXPECT_EQ(out[static_cast<std::size_t>(i)], global_i * (global_i + 1) / 2) << "rank=" << rank << " i=" << i;
    }
}

TEST(InclusiveScan, SingleElement) {
    // Each rank holds a single element equal to rank.
    // Inclusive scan: out[i] = 0+1+...+i = i*(i+1)/2.
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    rank = comm.rank();
    std::vector<int>             in   = {rank};
    std::vector<int>             out(1);

    dstl::inclusive_scan(in, out, std::plus<int>{}, comm);

    EXPECT_EQ(out[0], rank * (rank + 1) / 2);
}

TEST(InclusiveScan, ForwardRangeInput) {
    // Smoke test: std::list is a forward_range but not contiguous.
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    std::list<int>               in = {1, 2, 3};
    std::vector<int>             out(3);

    dstl::inclusive_scan(in, out, std::plus<int>{}, comm);

    int const prefix = 6 * comm.rank(); // each rank contributes 1+2+3 = 6
    EXPECT_EQ(out[0], prefix + 1);
    EXPECT_EQ(out[1], prefix + 3);
    EXPECT_EQ(out[2], prefix + 6);
}
