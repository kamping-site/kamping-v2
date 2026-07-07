// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <cstddef>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "alltoallv_test_common.hpp"
#include "dstl/dstl.hpp"
#include "kamping/v2/views.hpp"
#include "mpi/comm.hpp"
#include "mpi/environment.hpp"

using mpi::experimental::comm_view;
namespace views = kamping::v2::views;
using dstl_test::build_send;
using dstl_test::sorted;
using dstl_test::standard_alltoallv;
using dstl_test::world_rank;
using dstl_test::world_size;

// The flat alltoallv exists only for the thread_multiple model; the sequential / funneled exchange is
// just kamping::v2::alltoallv. All tests therefore require MPI_THREAD_MULTIPLE and skip otherwise.

// thread_multiple: the per-thread recv displacements reassemble each source's block contiguously in
// rank-then-thread order, so the result is element-identical to the flat alltoallv.
TEST(FlatAlltoallvTest, EqualsFlat) {
    if (mpi::experimental::environment::thread_level() < mpi::experimental::ThreadLevel::multiple) {
        GTEST_SKIP() << "runtime does not provide MPI_THREAD_MULTIPLE";
    }
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);
    std::vector<int> expected   = standard_alltoallv(data, counts, displs);

    dstl::thread_multiple_comm fc{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::alltoallv(
        data | views::with_counts(counts) | views::with_displs(displs),
        recv | views::auto_recv_v,
        fc
    );

    EXPECT_EQ(sorted(recv), sorted(expected)); // multiset always holds
    EXPECT_EQ(recv, expected);                 // and the layout is flat-identical here
}

// Owned (rvalue) recv buffer: the data lives in the returned result.
TEST(FlatAlltoallvTest, OwnedRecvBuffer) {
    if (mpi::experimental::environment::thread_level() < mpi::experimental::ThreadLevel::multiple) {
        GTEST_SKIP() << "runtime does not provide MPI_THREAD_MULTIPLE";
    }
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);
    std::vector<int> expected   = standard_alltoallv(data, counts, displs);

    dstl::thread_multiple_comm fc{comm_view{MPI_COMM_WORLD}};
    auto                       res = dstl::alltoallv(
        data | views::with_counts(counts) | views::with_displs(displs),
        std::vector<int>{} | views::auto_recv_v,
        fc
    );
    EXPECT_EQ(res.recv.underlying(), expected);
}

// Each rank sends exactly one element to each rank.
TEST(FlatAlltoallvTest, UniformSingleElement) {
    if (mpi::experimental::environment::thread_level() < mpi::experimental::ThreadLevel::multiple) {
        GTEST_SKIP() << "runtime does not provide MPI_THREAD_MULTIPLE";
    }
    int              rank = world_rank();
    int              size = world_size();
    std::vector<int> data(static_cast<std::size_t>(size), rank);
    std::vector<int> counts(static_cast<std::size_t>(size), 1);
    std::vector<int> displs(static_cast<std::size_t>(size));
    std::iota(displs.begin(), displs.end(), 0);

    dstl::thread_multiple_comm fc{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::alltoallv(data | views::with_counts(counts) | views::with_displs(displs), recv | views::auto_recv_v, fc);

    std::vector<int> expected(static_cast<std::size_t>(size));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(recv, expected);
}

// Deferred send buffer: sparse flatten_v() sends require set_comm_size() before counts/displs are
// read. Without the fix, ensure_flattened() would assert comm_size_.has_value() and abort.
TEST(FlatAlltoallvTest, DeferredSendBuf_SparseFlattenV) {
    if (mpi::experimental::environment::thread_level() < mpi::experimental::ThreadLevel::multiple) {
        GTEST_SKIP() << "runtime does not provide MPI_THREAD_MULTIPLE";
    }
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);
    std::vector<int> expected   = standard_alltoallv(data, counts, displs);

    // Same send pattern as build_send but packaged as sparse (destination, buffer) pairs in
    // reverse rank order — this triggers the sparse path in flatten_v_view that requires
    // set_comm_size() before it can lay out counts/displs/data.
    std::vector<std::pair<int, std::vector<int>>> per_dest;
    for (int j = size - 1; j >= 0; --j) {
        per_dest.emplace_back(j, std::vector<int>(static_cast<std::size_t>(rank + 1), rank * 10 + j));
    }

    dstl::thread_multiple_comm fc{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::alltoallv(per_dest | views::flatten_v(), recv | views::auto_recv_v, fc);

    EXPECT_EQ(recv, expected);
}

// Degenerate: every rank sends nothing.
TEST(FlatAlltoallvTest, AllEmpty) {
    if (mpi::experimental::environment::thread_level() < mpi::experimental::ThreadLevel::multiple) {
        GTEST_SKIP() << "runtime does not provide MPI_THREAD_MULTIPLE";
    }
    int              size = world_size();
    std::vector<int> data;
    std::vector<int> counts(static_cast<std::size_t>(size), 0);
    std::vector<int> displs(static_cast<std::size_t>(size), 0);

    dstl::thread_multiple_comm fc{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::alltoallv(data | views::with_counts(counts) | views::with_displs(displs), recv | views::auto_recv_v, fc);
    EXPECT_TRUE(recv.empty());
}
