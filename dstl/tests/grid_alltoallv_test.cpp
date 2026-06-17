// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dstl/dstl.hpp"
#include "kamping/v2/collectives/alltoallv.hpp"
#include "kamping/v2/views.hpp"
#include "kamping/v2/views/auto_recv_v.hpp"
#include "mpi/comm.hpp"

using namespace ::testing;
using mpi::experimental::comm_view;

namespace {

int world_size() {
    int s = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
}
int world_rank() {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

// rank i sends (i+1) copies of (i*10 + j) to rank j.
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> build_send(int rank, int size) {
    std::vector<int> counts(static_cast<std::size_t>(size));
    std::vector<int> data;
    for (int j = 0; j < size; ++j) {
        counts[static_cast<std::size_t>(j)] = rank + 1;
        for (int k = 0; k < rank + 1; ++k) {
            data.push_back(rank * 10 + j);
        }
    }
    std::vector<int> displs(static_cast<std::size_t>(size));
    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
    return {data, counts, displs};
}

// Oracle: the flat kamping::v2::alltoallv result on MPI_COMM_WORLD.
std::vector<int> flat_oracle(std::vector<int> const& data, std::vector<int> const& counts, std::vector<int> const& displs) {
    std::vector<int> recv;
    kamping::v2::alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv | kamping::v2::views::auto_recv_v
    );
    return recv;
}

std::vector<int> sorted(std::vector<int> v) {
    std::ranges::sort(v);
    return v;
}

} // namespace

// unordered: the recv buffer is multiset-equal to the flat alltoallv.
TEST(GridAlltoallvTest, UnorderedMultisetEqualsFlat) {
    int rank = world_rank();
    int size = world_size();
    auto [data, counts, displs] = build_send(rank, size);

    std::vector<int> expected = flat_oracle(data, counts, displs);

    dstl::grid_comm<dstl::sequential> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>                  recv;
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
    int rank = world_rank();
    int size = world_size();
    auto [data, counts, displs] = build_send(rank, size);

    std::vector<int> expected = flat_oracle(data, counts, displs);

    dstl::grid_comm<dstl::sequential> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>                  recv;
    dstl::alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv,
        grid,
        dstl::ordered_by_source{}
    );

    EXPECT_EQ(recv, expected);
}

// Owned (rvalue) recv buffer: the data lives in the returned result.
TEST(GridAlltoallvTest, OwnedRecvBuffer) {
    int rank = world_rank();
    int size = world_size();
    auto [data, counts, displs] = build_send(rank, size);

    std::vector<int> expected = flat_oracle(data, counts, displs);

    dstl::grid_comm<dstl::sequential> grid{comm_view{MPI_COMM_WORLD}};
    auto                              res = dstl::alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        std::vector<int>{},
        grid,
        dstl::ordered_by_source{}
    );

    EXPECT_EQ(res.recv, expected);
}

// Member-function entry point delegates to the free function.
TEST(GridAlltoallvTest, MemberEntryPoint) {
    int rank = world_rank();
    int size = world_size();
    auto [data, counts, displs] = build_send(rank, size);

    std::vector<int> expected = flat_oracle(data, counts, displs);

    dstl::grid_comm<dstl::sequential> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>                  recv;
    grid.alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv,
        dstl::ordered_by_source{}
    );

    EXPECT_EQ(recv, expected);
}

// Each rank sends exactly one element to each rank — same multiset as alltoall.
TEST(GridAlltoallvTest, UniformSingleElement) {
    int              rank = world_rank();
    int              size = world_size();
    std::vector<int> data(static_cast<std::size_t>(size), rank);
    std::vector<int> counts(static_cast<std::size_t>(size), 1);
    std::vector<int> displs(static_cast<std::size_t>(size));
    std::iota(displs.begin(), displs.end(), 0);

    dstl::grid_comm<dstl::sequential> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>                  recv;
    dstl::alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv,
        grid,
        dstl::ordered_by_source{}
    );

    std::vector<int> expected(static_cast<std::size_t>(size));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(recv, expected);
}

// Degenerate: every rank sends nothing.
TEST(GridAlltoallvTest, AllEmpty) {
    int                               size = world_size();
    std::vector<int>                  data;
    std::vector<int>                  counts(static_cast<std::size_t>(size), 0);
    std::vector<int>                  displs(static_cast<std::size_t>(size), 0);
    dstl::grid_comm<dstl::sequential> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>                  recv;
    dstl::alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv,
        grid
    );
    EXPECT_TRUE(recv.empty());
}

// Explicit non-default factorizations produce the same result (exercises different k / dims).
TEST(GridAlltoallvTest, ExplicitFactorizations) {
    int rank = world_rank();
    int size = world_size();
    auto [data, counts, displs] = build_send(rank, size);
    std::vector<int> expected   = flat_oracle(data, counts, displs);

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
        dstl::grid_comm<dstl::sequential> grid{comm_view{MPI_COMM_WORLD}, std::span<std::size_t const>{dims}};
        std::vector<int>                  recv;
        dstl::alltoallv(
            data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
            recv,
            grid,
            dstl::ordered_by_source{}
        );
        EXPECT_EQ(recv, expected) << "factorization with k=" << dims.size();
    }
}
