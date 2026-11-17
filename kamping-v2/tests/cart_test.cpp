// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <array>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/cart.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/p2p/sendrecv.hpp"
#include "kamping/v2/views.hpp"

namespace v2 = kamping::v2;

static_assert(std::is_aggregate_v<v2::cart_dim>);

TEST(V2CartTest, CreateWithAutoDims) {
    v2::comm_view const world = v2::comm_world();
    v2::comm const      grid  = v2::cart_create(world, {{v2::auto_dim, true}, {v2::auto_dim, true}});
    EXPECT_EQ(grid.topology(), v2::topology::cart);
    EXPECT_EQ(grid.size(), world.size());
    EXPECT_EQ(grid.rank(), world.rank()); // reorder is false if left out

    auto const dims     = v2::cart_dims(grid);
    auto const expected = v2::dims_create<2>(world.size());
    ASSERT_EQ(dims.size(), 2u);
    EXPECT_EQ(dims[0].procs, expected[0]);
    EXPECT_EQ(dims[1].procs, expected[1]);
    EXPECT_TRUE(dims[0].periodic && dims[1].periodic);

    auto const [rows, cols] = v2::cart_dims<2>(grid);
    EXPECT_EQ(rows.procs, expected[0]);
    EXPECT_EQ(cols.procs, expected[1]);
    EXPECT_TRUE(rows.periodic && cols.periodic);
}

TEST(V2CartTest, CreateWithFixedAndAutoDims) {
    v2::comm_view const world = v2::comm_world();
    // One column fixed to the full size, the rows chosen: a single row.
    std::vector<v2::cart_dim> const dims{{.procs = v2::auto_dim}, {.procs = world.size(), .periodic = true}};
    v2::comm const                  grid = v2::cart_create(world, dims, true);
    auto const                      got  = v2::cart_dims(grid);
    EXPECT_EQ(got[0].procs, 1);
    EXPECT_FALSE(got[0].periodic);
    EXPECT_EQ(got[1].procs, world.size());
    EXPECT_TRUE(got[1].periodic);
}

TEST(V2CartTest, RingWithPositions) {
    // A ring along dimension 1 of a periodic grid, with the positions passed to v2::sendrecv as ranks.
    v2::comm const grid      = v2::cart_create(v2::comm_world(), {{v2::auto_dim, true}, {v2::auto_dim, true}});
    auto const [left, right] = grid.cart_shift(1, 1);
    std::vector<int> const sbuf{grid.rank()};
    std::vector<int>       rbuf(1, -1);
    v2::sendrecv(sbuf, right, rbuf, left, grid);
    EXPECT_EQ(rbuf[0], left.value());
}

TEST(V2CartTest, RowAndColumnCommunicators) {
    v2::comm const grid = v2::cart_create(v2::comm_world(), {{v2::auto_dim}, {v2::auto_dim}});
    auto const     dims = v2::cart_dims(grid);
    EXPECT_EQ(grid.cart_sub(1).size(), dims[1].procs);
    EXPECT_EQ(v2::cart_sub(grid, 0).size(), dims[0].procs);
}
