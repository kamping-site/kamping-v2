// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <functional>
#include <list>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dstl/algorithm/exclusive_scan.hpp"
#include "dstl/algorithm/inclusive_scan.hpp"
#include "dstl/algorithm/reduce.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/datatype.hpp"
#include "kamping/v2/ops.hpp"
#include "kamping/v2/views/with_type_view.hpp"

// ── helpers ───────────────────────────────────────────────────────────────────

namespace {
std::vector<int> iota_vec(int start, int n) {
    std::vector<int> v(static_cast<std::size_t>(n));
    std::iota(v.begin(), v.end(), start);
    return v;
}
} // namespace

// ── op constraint: the operation must combine values locally and through MPI ──

namespace {
template <typename Op>
concept reducible_with = requires(std::vector<int> const& v, Op const& op) { dstl::reduce(v, 0, op); };
template <typename Op>
concept scannable_with = requires(std::vector<int> const& v, std::vector<int>& out, Op const& op) {
    dstl::inclusive_scan(v, out, op);
    dstl::exclusive_scan(v, out, 0, op);
};
inline auto const captured = 1;
using capturing_lambda     = decltype([c = captured](int a, int b) { return a + b + c; });
using sum_op               = decltype(kamping::v2::make_op<int>(std::plus<>{}));
} // namespace

static_assert(reducible_with<std::plus<>>);
static_assert(reducible_with<kamping::v2::ops::max<>>);
static_assert(reducible_with<sum_op>);
static_assert(reducible_with<kamping::v2::op>); // callable on int through MPI_Reduce_local
static_assert(reducible_with<kamping::v2::op_view>);
static_assert(scannable_with<std::plus<>>);
static_assert(scannable_with<sum_op>);
static_assert(!reducible_with<std::minus<>>);     // no predefined operation
static_assert(!reducible_with<capturing_lambda>); // not an MPI operation
static_assert(reducible_with<MPI_Op>);            // applied locally through MPI_Reduce_local
static_assert(!scannable_with<capturing_lambda>);

// ─────────────────────────────────────────────────────────────────────────────
// dstl::reduce
// ─────────────────────────────────────────────────────────────────────────────

TEST(Reduce, Sum) {
    // Each rank holds [rank*4, rank*4+1, rank*4+2, rank*4+3].
    // Global sum = 0+1+...+(4p-1) = 4p*(4p-1)/2.
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    int const                    p    = comm.size();
    int const                    n    = 4;
    auto const                   vec  = iota_vec(comm.rank() * n, n);

    int const result   = dstl::reduce(vec, 0, std::plus<int>{}, comm);
    int const expected = n * p * (n * p - 1) / 2;
    EXPECT_EQ(result, expected);
}

TEST(Reduce, WithInit) {
    // init is added to the global sum.
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    int const                    n    = 2;
    auto const                   vec  = iota_vec(comm.rank() * n, n);

    int const total  = n * comm.size() * (n * comm.size() - 1) / 2;
    int const result = dstl::reduce(vec, 100, std::plus<int>{}, comm);
    EXPECT_EQ(result, 100 + total);
}

TEST(Reduce, EmptyLocal) {
    // All ranks have empty ranges — result should equal init.
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    std::vector<int> const       empty;

    int const result = dstl::reduce(empty, 42, std::plus<int>{}, comm);
    EXPECT_EQ(result, 42);
}

TEST(Reduce, MakeOpHandle) {
    // ops from make_op are not copyable; they must be accepted by reference.
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    int const                    n    = 4;
    auto const                   vec  = iota_vec(comm.rank() * n, n);

    auto      op       = kamping::v2::make_op<int>([](int a, int b) { return a + b; }, kamping::v2::commutative);
    int const result   = dstl::reduce(vec, 0, op, comm);
    int const expected = n * comm.size() * (n * comm.size() - 1) / 2;
    EXPECT_EQ(result, expected);
}

TEST(Reduce, ForwardRangeInput) {
    // Smoke test: input is a std::list (forward_range, not contiguous).
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    std::list<int> const         vals = {1, 2, 3};

    int const result = dstl::reduce(vals, 0, std::plus<int>{}, comm);
    EXPECT_EQ(result, 6 * comm.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// dstl::reduce_to_root
// ─────────────────────────────────────────────────────────────────────────────

TEST(ReduceToRoot, Sum) {
    // Root receives the global sum; all other ranks get nullopt.
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    int const                    n    = 4;
    auto const                   vec  = iota_vec(comm.rank() * n, n);
    int const                    p    = comm.size();

    auto const result   = dstl::reduce_to_root(vec, 0, std::plus<int>{}, 0, comm);
    int const  expected = n * p * (n * p - 1) / 2;

    if (comm.rank() == 0) {
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(*result, expected);
    } else {
        EXPECT_FALSE(result.has_value());
    }
}

TEST(ReduceToRoot, MakeOpHandle) {
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    int const                    n    = 4;
    auto const                   vec  = iota_vec(comm.rank() * n, n);

    auto       op       = kamping::v2::make_op<int>([](int a, int b) { return a + b; }, kamping::v2::commutative);
    auto const result   = dstl::reduce_to_root(vec, 0, op, 0, comm);
    int const  expected = n * comm.size() * (n * comm.size() - 1) / 2;

    if (comm.rank() == 0) {
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(*result, expected);
    } else {
        EXPECT_FALSE(result.has_value());
    }
}

TEST(ReduceToRoot, WithInit) {
    // init is added once to the global sum, only on root.
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    int const                    n    = 2;
    auto const                   vec  = iota_vec(comm.rank() * n, n);
    int const                    p    = comm.size();

    auto const result = dstl::reduce_to_root(vec, 100, std::plus<int>{}, 0, comm);
    int const  total  = n * p * (n * p - 1) / 2;

    if (comm.rank() == 0) {
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(*result, 100 + total);
    } else {
        EXPECT_FALSE(result.has_value());
    }
}

TEST(ReduceToRoot, NonDefaultRoot) {
    // Last rank is root.
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    int const                    root = comm.size() - 1;
    auto const                   vec  = iota_vec(comm.rank() * 2, 2);
    int const                    p    = comm.size();

    auto const result   = dstl::reduce_to_root(vec, 0, std::plus<int>{}, root, comm);
    int const  expected = 2 * p * (2 * p - 1) / 2;

    if (comm.rank() == root) {
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(*result, expected);
    } else {
        EXPECT_FALSE(result.has_value());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// dstl::exclusive_scan
// ─────────────────────────────────────────────────────────────────────────────

TEST(ExclusiveScan, UniformChunks) {
    // Each rank holds [rank*4, ..., rank*4+3].
    // Global exclusive scan: out[i] = 0+1+...+(global_i - 1).
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
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

TEST(ExclusiveScan, MakeOpHandle) {
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    int const                    n    = 4;
    int const                    base = comm.rank() * n;
    auto const                   in   = iota_vec(base, n);
    std::vector<int>             out(static_cast<std::size_t>(n), 0);

    auto op = kamping::v2::make_op<int>([](int a, int b) { return a + b; }, kamping::v2::commutative);
    dstl::exclusive_scan(in, out, 0, op, comm);

    for (int i = 0; i < n; ++i) {
        int const global_i = base + i;
        EXPECT_EQ(out[static_cast<std::size_t>(i)], global_i * (global_i - 1) / 2);
    }
}

TEST(ExclusiveScan, NonCommutativeOp) {
    // "Keep the right operand" is associative but not commutative: element i of the exclusive scan is element i - 1
    // of the global sequence, so the local total must be combined in order.
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    int const                    n    = 5;
    int const                    base = comm.rank() * n;
    auto const                   in   = iota_vec(base, n);
    std::vector<int>             out(static_cast<std::size_t>(n), 0);

    auto keep_right = kamping::v2::make_op<int>([](int /* a */, int b) { return b; });
    dstl::exclusive_scan(in, out, -1, keep_right, comm);

    for (int i = 0; i < n; ++i) {
        int const global_i = base + i;
        EXPECT_EQ(out[static_cast<std::size_t>(i)], global_i == 0 ? -1 : global_i - 1);
    }
}

TEST(ExclusiveScan, SingleElement) {
    // Each rank holds a single element equal to rank.
    // Exclusive scan: out[i] = 0+1+...+(i-1) = i*(i-1)/2.
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    int const                    rank = comm.rank();
    std::vector<int>             in   = {rank};
    std::vector<int>             out(1);

    dstl::exclusive_scan(in, out, 0, std::plus<int>{}, comm);

    EXPECT_EQ(out[0], rank * (rank - 1) / 2);
}

TEST(ExclusiveScan, WithInit) {
    // Non-zero init shifts the entire output.
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    int const                    rank = comm.rank();
    std::vector<int>             in   = {1};
    std::vector<int>             out(1);

    dstl::exclusive_scan(in, out, 10, std::plus<int>{}, comm);

    // out[0] = init + sum of elements on ranks 0..rank-1 = 10 + rank.
    EXPECT_EQ(out[0], 10 + rank);
}

TEST(ExclusiveScan, ForwardRangeInput) {
    // Smoke test: std::list is a forward_range but not contiguous.
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    std::list<int>               in   = {1, 2, 3};
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
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
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

TEST(InclusiveScan, MakeOpHandle) {
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    int const                    n    = 4;
    int const                    base = comm.rank() * n;
    auto const                   in   = iota_vec(base, n);
    std::vector<int>             out(static_cast<std::size_t>(n), 0);

    auto op = kamping::v2::make_op<int>([](int a, int b) { return a + b; }, kamping::v2::commutative);
    dstl::inclusive_scan(in, out, op, comm);

    for (int i = 0; i < n; ++i) {
        int const global_i = base + i;
        EXPECT_EQ(out[static_cast<std::size_t>(i)], global_i * (global_i + 1) / 2);
    }
}

TEST(InclusiveScan, SingleElement) {
    // Each rank holds a single element equal to rank.
    // Inclusive scan: out[i] = 0+1+...+i = i*(i+1)/2.
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    int const                    rank = comm.rank();
    std::vector<int>             in   = {rank};
    std::vector<int>             out(1);

    dstl::inclusive_scan(in, out, std::plus<int>{}, comm);

    EXPECT_EQ(out[0], rank * (rank + 1) / 2);
}

TEST(InclusiveScan, ForwardRangeInput) {
    // Smoke test: std::list is a forward_range but not contiguous.
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    std::list<int>               in   = {1, 2, 3};
    std::vector<int>             out(3);

    dstl::inclusive_scan(in, out, std::plus<int>{}, comm);

    int const prefix = 6 * comm.rank(); // each rank contributes 1+2+3 = 6
    EXPECT_EQ(out[0], prefix + 1);
    EXPECT_EQ(out[1], prefix + 3);
    EXPECT_EQ(out[2], prefix + 6);
}

// ─────────────────────────────────────────────────────────────────────────────
// Operations that are only MPI operations, applied locally through MPI_Reduce_local
// ─────────────────────────────────────────────────────────────────────────────

namespace {
struct int_pair {
    int a;
    int b;
};

// Adds int_pairs component-wise.
void add_pairs(void* invec, void* inoutvec, int* len, MPI_Datatype* /* datatype */) {
    auto const* in    = static_cast<int_pair const*>(invec);
    auto*       inout = static_cast<int_pair*>(inoutvec);
    for (int i = 0; i < *len; ++i) {
        inout[i].a += in[i].a;
        inout[i].b += in[i].b;
    }
}
} // namespace

TEST(Reduce, RawMpiOp) {
    kamping::v2::comm_view const comm = kamping::v2::comm_world();
    int const                    n    = 4;
    auto const                   vec  = iota_vec(comm.rank() * n, n);

    int const result   = dstl::reduce(vec, 0, MPI_SUM, comm);
    int const expected = n * comm.size() * (n * comm.size() - 1) / 2;
    EXPECT_EQ(result, expected);
}

TEST(InclusiveScan, CallbackOpOnDerivedType) {
    kamping::v2::comm_view const comm      = kamping::v2::comm_world();
    auto const                   pair_type = kamping::v2::type_contiguous(2, MPI_INT).commit();
    auto const                   add       = kamping::v2::make_op(&add_pairs, kamping::v2::commutative);

    // Each rank holds three pairs {1, rank}; the inclusive scan counts the pairs and sums the ranks before them.
    std::vector<int_pair> const in(3, int_pair{1, comm.rank()});
    std::vector<int_pair>       out(3);
    dstl::inclusive_scan(
        in | kamping::v2::views::with_type(pair_type),
        out | kamping::v2::views::with_type(pair_type),
        add,
        comm
    );

    int const ranks_before = comm.rank() * (comm.rank() - 1) / 2;
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(out[static_cast<std::size_t>(i)].a, 3 * comm.rank() + i + 1);
        EXPECT_EQ(out[static_cast<std::size_t>(i)].b, 3 * ranks_before + (i + 1) * comm.rank());
    }
}
