// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dstl/dstl.hpp"
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
} // namespace

// The default factoring (dims_create{2}) yields a complete grid whose factors multiply to p.
TEST(GridCommTest, DefaultDimsAreComplete) {
    dstl::grid_comm<dstl::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::size_t                       product = 1;
    for (std::size_t i = 0; i < grid.num_dims(); ++i) {
        product *= grid.dim_size(i);
    }
    EXPECT_EQ(product, static_cast<std::size_t>(world_size()));
    EXPECT_TRUE(grid.is_complete());
}

// coords() and rank_of() are inverses for every global rank.
TEST(GridCommTest, CoordRoundTrip) {
    dstl::grid_comm<dstl::seq> grid{comm_view{MPI_COMM_WORLD}};
    for (int r = 0; r < world_size(); ++r) {
        auto coords = grid.coords(r);
        EXPECT_EQ(grid.rank_of(coords), r) << "round-trip failed for rank " << r;
    }
}

// For a complete grid, the calling rank's position inside subcommunicator i equals coordinate i, and
// the subcommunicator has the expected size s_i.
TEST(GridCommTest, SubcommRankMatchesCoordinate) {
    dstl::grid_comm<dstl::seq> grid{comm_view{MPI_COMM_WORLD}};
    auto                              coords = grid.coords(world_rank());
    for (std::size_t i = 0; i < grid.num_dims(); ++i) {
        EXPECT_EQ(static_cast<std::size_t>(grid.subcomm(i).size()), grid.dim_size(i));
        EXPECT_EQ(static_cast<std::size_t>(grid.subcomm(i).rank()), coords[i]) << "dimension " << i;
    }
}

// Explicit dimensions are honored verbatim (here: a 1-D grid degenerates to the flat communicator).
TEST(GridCommTest, ExplicitOneDimension) {
    std::vector<std::size_t>          dims = {static_cast<std::size_t>(world_size())};
    dstl::grid_comm<dstl::seq> grid{comm_view{MPI_COMM_WORLD}, std::span<std::size_t const>{dims}};
    EXPECT_EQ(grid.num_dims(), 1u);
    EXPECT_EQ(grid.dim_size(0), static_cast<std::size_t>(world_size()));
    EXPECT_EQ(static_cast<std::size_t>(grid.subcomm(0).size()), static_cast<std::size_t>(world_size()));
}

// A 2x2 grid on four ranks: verify the explicit mixed-radix layout. The last dimension (stride 1)
// groups consecutive ranks; dimension 0 (stride 2) is the remote one.
TEST(GridCommTest, ExplicitTwoByTwo) {
    if (world_size() != 4) {
        GTEST_SKIP() << "requires exactly 4 ranks";
    }
    std::vector<std::size_t>          dims = {2, 2};
    dstl::grid_comm<dstl::seq> grid{comm_view{MPI_COMM_WORLD}, std::span<std::size_t const>{dims}};
    // Row-major (c0 most significant): rank 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1).
    EXPECT_EQ(grid.coords(0), (std::vector<std::size_t>{0, 0}));
    EXPECT_EQ(grid.coords(1), (std::vector<std::size_t>{0, 1}));
    EXPECT_EQ(grid.coords(2), (std::vector<std::size_t>{1, 0}));
    EXPECT_EQ(grid.coords(3), (std::vector<std::size_t>{1, 1}));
    // subcomm 1 has stride 1 → consecutive ranks {0,1} / {2,3}; subcomm 0 has stride 2 → {0,2}/{1,3}.
    EXPECT_EQ(grid.stride(0), 2u);
    EXPECT_EQ(grid.stride(1), 1u);
}
