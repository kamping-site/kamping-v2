// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <utility>
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

    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::grid_alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv | views::resize, // opt into automatic resizing (bare buffers are assumed pre-sized)
        grid,
        dstl::layout::unordered{}
    );

    EXPECT_EQ(sorted(recv), sorted(expected));
}

// ordered_by_source: the recv buffer is element-identical to the flat alltoallv.
TEST(GridAlltoallvTest, OrderedEqualsFlatExactly) {
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);

    std::vector<int> expected = standard_alltoallv(data, counts, displs);

    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::grid_alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv | views::auto_recv_v,
        grid,
        dstl::layout::ordered_by_source{}
    );

    EXPECT_EQ(recv, expected);
}

// Owned (rvalue) recv buffer: the data lives in the returned result.
TEST(GridAlltoallvTest, OwnedRecvBuffer) {
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);

    std::vector<int> expected = standard_alltoallv(data, counts, displs);

    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    auto                       res = dstl::grid_alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        std::vector<int>{} | views::auto_recv_v,
        grid,
        dstl::layout::ordered_by_source{}
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

    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::grid_alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv | views::auto_recv_v,
        grid,
        dstl::layout::ordered_by_source{}
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
    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::grid_alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv,
        grid
    );
    EXPECT_TRUE(recv.empty());
}

// Opt-in resize: a PRE-SIZED bare recv buffer (no views::resize) is written into as-is.
TEST(GridAlltoallvTest, PreSizedBareRecvBuffer) {
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);

    std::vector<int> expected = standard_alltoallv(data, counts, displs);

    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv(expected.size()); // caller pre-sizes; no views::resize
    dstl::grid_alltoallv(data | views::with_counts(counts) | views::with_displs(displs), recv, grid, dstl::layout::unordered{});

    EXPECT_EQ(sorted(recv), sorted(expected));
}

// Headline of the relaxed contract: send and recv may use DIFFERENT element types / datatypes as long as
// the MPI type signatures match (exactly like a plain MPI_Alltoallv). Send a struct with a gap whose
// datatype spans only two of three members; receive into a packed two-int struct. The gap is never
// transmitted; the two payload ints arrive intact.
TEST(GridAlltoallvTest, MixedGappedSendPackedRecv) {
    int rank = world_rank();
    int size = world_size();

    struct SendS {
        int a;
        int gap;
        int b;
    };
    struct RecvS {
        int a;
        int b;
    };

    // dt_send: blocks {a@offsetof(a), b@offsetof(b)} of MPI_INT, extent resized to sizeof(SendS).
    MPI_Datatype dt_send;
    {
        int          blocklen[2] = {1, 1};
        MPI_Aint     disp[2]     = {offsetof(SendS, a), offsetof(SendS, b)};
        MPI_Datatype types[2]    = {MPI_INT, MPI_INT};
        MPI_Datatype tmp;
        MPI_Type_create_struct(2, blocklen, disp, types, &tmp);
        MPI_Type_create_resized(tmp, 0, static_cast<MPI_Aint>(sizeof(SendS)), &dt_send);
        MPI_Type_commit(&dt_send);
        MPI_Type_free(&tmp);
    }
    // dt_recv: two contiguous ints, extent sizeof(RecvS). Same signature (INT, INT) as dt_send.
    MPI_Datatype dt_recv;
    MPI_Type_contiguous(2, MPI_INT, &dt_recv);
    MPI_Type_commit(&dt_recv);

    // Each rank sends exactly one SendS to every rank d: a = rank*100 + d, b = rank*1000 + d, gap = -1.
    std::vector<SendS> data(static_cast<std::size_t>(size));
    for (int d = 0; d < size; ++d) {
        data[static_cast<std::size_t>(d)] = SendS{rank * 100 + d, -1, rank * 1000 + d};
    }
    std::vector<int> counts(static_cast<std::size_t>(size), 1);
    std::vector<int> displs(static_cast<std::size_t>(size));
    std::iota(displs.begin(), displs.end(), 0);

    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<RecvS>         recv;
    dstl::grid_alltoallv(
        data | views::with_type(dt_send) | views::with_counts(counts) | views::with_displs(displs),
        recv | views::with_type(dt_recv) | views::resize,
        grid,
        dstl::layout::unordered{}
    );

    // Rank d receives one element from each source r: (a, b) = (r*100 + d, r*1000 + d), gap dropped.
    std::vector<std::pair<int, int>> got;
    for (auto const& e: recv) {
        got.emplace_back(e.a, e.b);
    }
    std::vector<std::pair<int, int>> want;
    for (int r = 0; r < size; ++r) {
        want.emplace_back(r * 100 + rank, r * 1000 + rank);
    }
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    EXPECT_EQ(got, want);

    MPI_Type_free(&dt_send);
    MPI_Type_free(&dt_recv);
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
        dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}, std::span<std::size_t const>{dims}};
        std::vector<int>           recv;
        dstl::grid_alltoallv(
            data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
            recv | views::auto_recv_v,
            grid,
            dstl::layout::ordered_by_source{}
        );
        EXPECT_EQ(recv, expected) << "factorization with k=" << dims.size();
    }
}
