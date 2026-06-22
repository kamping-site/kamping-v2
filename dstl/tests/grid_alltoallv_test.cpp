// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "alltoallv_test_common.hpp"
#include "dstl/dstl.hpp"
#include "kamping/v2/views.hpp"
#include "mpi/comm.hpp"

using namespace ::testing;
using mpi::experimental::comm_view;
namespace views = kamping::v2::views;
using dstl_test::build_send;
using dstl_test::sorted;
using dstl_test::standard_alltoallv;
using dstl_test::world_rank;
using dstl_test::world_size;

// unordered: the recv buffer is multiset-equal to the flat alltoallv.
TEST(GridAlltoallvTest, UnorderedMultisetEqualsFlat) {
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);

    std::vector<int> expected = standard_alltoallv(data, counts, displs);

    dstl::grid_comm<dstl::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv,
        grid,
        dstl::unordered{}
    );

    EXPECT_EQ(sorted(recv), sorted(expected));
}

// ordered_by_source: the recv buffer is element-identical to the flat alltoallv.
TEST(GridAlltoallvTest, OrderedEqualsFlatExactly) {
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);

    std::vector<int> expected = standard_alltoallv(data, counts, displs);

    dstl::grid_comm<dstl::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv | views::auto_recv_v,
        grid,
        dstl::ordered_by_source{}
    );

    EXPECT_EQ(recv, expected);
}

// Owned (rvalue) recv buffer: the data lives in the returned result.
TEST(GridAlltoallvTest, OwnedRecvBuffer) {
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);

    std::vector<int> expected = standard_alltoallv(data, counts, displs);

    dstl::grid_comm<dstl::seq> grid{comm_view{MPI_COMM_WORLD}};
    auto                       res = dstl::alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        std::vector<int>{} | views::auto_recv_v,
        grid,
        dstl::ordered_by_source{}
    );

    EXPECT_EQ(res.recv.underlying(), expected);
}

// Each rank sends exactly one element to each rank — same multiset as alltoall.
TEST(GridAlltoallvTest, UniformSingleElement) {
    int              rank = world_rank();
    int              size = world_size();
    std::vector<int> data(static_cast<std::size_t>(size), rank);
    std::vector<int> counts(static_cast<std::size_t>(size), 1);
    std::vector<int> displs(static_cast<std::size_t>(size));
    std::iota(displs.begin(), displs.end(), 0);

    dstl::grid_comm<dstl::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv | views::auto_recv_v,
        grid,
        dstl::ordered_by_source{}
    );

    std::vector<int> expected(static_cast<std::size_t>(size));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(recv, expected);
}

// Degenerate: every rank sends nothing.
TEST(GridAlltoallvTest, AllEmpty) {
    int                        size = world_size();
    std::vector<int>           data;
    std::vector<int>           counts(static_cast<std::size_t>(size), 0);
    std::vector<int>           displs(static_cast<std::size_t>(size), 0);
    dstl::grid_comm<dstl::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv,
        grid
    );
    EXPECT_TRUE(recv.empty());
}

// Explicit non-default factorizations produce the same result (exercises different k / dims).
TEST(GridAlltoallvTest, ExplicitFactorizations) {
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);
    std::vector<int> expected   = standard_alltoallv(data, counts, displs);

    // A 1-D grid (flat), and — when 4 ranks — a 2x2 grid.
    std::vector<std::vector<std::size_t>> factorings = {{static_cast<std::size_t>(size)}};
    if (size == 4) {
        factorings.push_back({2, 2});
    }
    if (size == 8) {
        factorings.push_back({2, 4});
        factorings.push_back({2, 2, 2});
    }

    for (auto const& dims: factorings) {
        dstl::grid_comm<dstl::seq> grid{comm_view{MPI_COMM_WORLD}, std::span<std::size_t const>{dims}};
        std::vector<int>           recv;
        dstl::alltoallv(
            data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
            recv | views::auto_recv_v,
            grid,
            dstl::ordered_by_source{}
        );
        EXPECT_EQ(recv, expected) << "factorization with k=" << dims.size();
    }
}
