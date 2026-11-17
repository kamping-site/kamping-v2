// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <array>
#include <span>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/cart.hpp"
#include "mpi/comm.hpp"
#include "mpi/handle.hpp"
#include "mpi/special_values.hpp"

using namespace mpi::experimental;

namespace {
/// A balanced, periodic two-dimensional grid over MPI_COMM_WORLD.
comm periodic_grid() {
    auto const dims = dims_create<2>(comm_world().size());
    return cart_create(MPI_COMM_WORLD, dims, std::array{true, true}, false);
}

/// The same grid, but non-periodic.
comm open_grid() {
    auto const dims = dims_create<2>(comm_world().size());
    return cart_create(MPI_COMM_WORLD, dims, std::array{false, false}, false);
}
} // namespace

// A position is usable wherever a rank is expected, also as a destination.
static_assert(destination<cart_position> && sender<cart_position>);
static_assert(flag_array<std::array<bool, 2>> && flag_array<std::vector<bool>> && flag_array<std::vector<int>>);
static_assert(!flag_array<std::vector<long>>);

// ── dims_create / topology ─────────────────────────────────────────────────────────────────────────────────────────

TEST(CartTest, DimsCreate) {
    EXPECT_EQ(dims_create<2>(12), (std::array{4, 3}));
    EXPECT_EQ(dims_create(12, std::array{auto_dim, 2}), (std::array{6, 2}));
    std::vector<int> dims{auto_dim, auto_dim, 2};
    dims_create(8, dims);
    EXPECT_EQ(dims, (std::vector{2, 2, 2}));
}

TEST(CartTest, Topology) {
    EXPECT_EQ(topo_test(MPI_COMM_WORLD), topology::none);
    EXPECT_EQ(comm_world().topology(), topology::none);
    comm const grid = periodic_grid();
    EXPECT_EQ(topo_test(grid), topology::cart);
    EXPECT_EQ(grid.topology(), topology::cart);
}

// ── cart_create and queries ────────────────────────────────────────────────────────────────────────────────────────

TEST(CartTest, CreateAndQuery) {
    comm_view const world = comm_world();
    auto const      dims  = dims_create<2>(world.size());
    comm const      grid  = cart_create(world, dims, std::vector<int>{1, 0}, false);
    EXPECT_EQ(grid.size(), world.size());
    EXPECT_EQ(grid.rank(), world.rank()); // no reordering
    EXPECT_EQ(cartdim(grid), 2);
    EXPECT_EQ(grid.cartdim(), 2);

    std::array<int, 2> got_dims{}, periods{}, coords{};
    cart_get(grid, got_dims, periods, coords);
    EXPECT_EQ(got_dims, dims);
    EXPECT_EQ(periods, (std::array{1, 0}));

    // Row-major: the last dimension varies fastest.
    EXPECT_EQ(coords, (std::array{world.rank() / dims[1], world.rank() % dims[1]}));
    EXPECT_EQ(grid.cart_coords<2>(), coords);
    EXPECT_EQ(cart_coords(grid), (std::vector<int>{coords[0], coords[1]}));
    EXPECT_EQ(cart_rank(grid, coords), world.rank());
    EXPECT_EQ(grid.cart_rank({coords[0], coords[1]}), world.rank());
    EXPECT_EQ(cart_coords<2>(grid, 0), (std::array{0, 0}));
}

TEST(CartTest, CreateIntoRawHandleWithLeftOverProcesses) {
    // A one-process grid: every other process gets MPI_COMM_NULL.
    MPI_Comm   raw  = MPI_COMM_NULL;
    std::array dims = {1};
    cart_create(MPI_COMM_WORLD, dims, std::vector<bool>{false}, false, raw);
    if (comm_world().rank() == 0) {
        EXPECT_NE(raw, MPI_COMM_NULL);
        MPI_Comm_free(&raw);
    } else {
        EXPECT_EQ(raw, MPI_COMM_NULL);
    }
}

// ── cart_shift / cart_position ─────────────────────────────────────────────────────────────────────────────────────

TEST(CartTest, ShiftInPeriodicGrid) {
    comm const grid          = periodic_grid();
    auto const last          = cart_coords<2>(grid, grid.size() - 1);
    int const  rows          = last[0] + 1;
    int const  cols          = last[1] + 1;
    auto const [row, col]    = grid.cart_coords<2>();
    auto const [left, right] = grid.cart_shift(1, 1);
    EXPECT_FALSE(right.is_proc_null());
    EXPECT_EQ(right.value(), cart_rank(grid, {row, (col + 1) % cols}));
    EXPECT_EQ(left.value(), cart_rank(grid, {row, (col - 1 + cols) % cols}));

    // Positions agree with MPI_Cart_shift and wrap around in periodic dimensions.
    auto const me = grid.cart_position();
    EXPECT_EQ(me.value(), grid.rank());
    EXPECT_EQ(me.shift(1, 1), right);
    EXPECT_EQ(me.shift(1, -1), left);
    EXPECT_EQ(me.shift(0, rows), me);
    EXPECT_EQ(me.shift(1, 1).coords()[1], (col + 1) % cols);
}

TEST(CartTest, ShiftInNonPeriodicGrid) {
    comm const grid          = open_grid();
    auto const last          = cart_coords<2>(grid, grid.size() - 1);
    int const  cols          = last[1] + 1;
    auto const [row, col]    = grid.cart_coords<2>();
    auto const [left, right] = cart_shift(grid, 1, 1);
    EXPECT_EQ(left.is_proc_null(), col == 0);
    EXPECT_EQ(right.is_proc_null(), col == cols - 1);
    if (!right.is_proc_null()) {
        EXPECT_EQ(right.value(), cart_rank(grid, {row, col + 1}));
    } else {
        EXPECT_EQ(right.value(), MPI_PROC_NULL);
    }

    // Beyond the ends, a position keeps its coordinates, so shifting back returns into the grid.
    auto const me      = grid.cart_position();
    auto const outside = me.shift(1, cols);
    EXPECT_TRUE(outside.is_proc_null());
    EXPECT_EQ(outside.coords()[1], col + cols);
    EXPECT_EQ(outside.shift(1, -cols), me);
    EXPECT_EQ(right.shift(1, -1), me);
}

TEST(CartTest, PositionAsRankInCommunication) {
    // A ring along dimension 1 of the periodic grid: every process passes its rank to the right.
    comm const grid          = periodic_grid();
    auto const [left, right] = grid.cart_shift(1, 1);
    int value                = grid.rank();
    int err =
        MPI_Sendrecv_replace(&value, 1, MPI_INT, to_rank(right), 0, to_rank(left), 0, handle(grid), MPI_STATUS_IGNORE);
    ASSERT_EQ(err, MPI_SUCCESS);
    EXPECT_EQ(value, left.value());
}

TEST(CartTest, DiagonalNeighbor) {
    comm const grid = periodic_grid();
    auto const me   = grid.cart_position();
    auto const diag = me.shift(0, 1).shift(1, 1);
    auto const c    = me.coords();
    auto const last = cart_coords<2>(grid, grid.size() - 1);
    EXPECT_EQ(diag.value(), cart_rank(grid, {(c[0] + 1) % (last[0] + 1), (c[1] + 1) % (last[1] + 1)}));
    EXPECT_EQ(grid.cart_position(diag.value()), diag);
}

// ── cart_sub ───────────────────────────────────────────────────────────────────────────────────────────────────────

TEST(CartTest, Sub) {
    comm const grid = periodic_grid();
    auto const last = cart_coords<2>(grid, grid.size() - 1);
    int const  rows = last[0] + 1;
    int const  cols = last[1] + 1;

    comm const row = grid.cart_sub(1); // the processes of my row
    EXPECT_EQ(row.size(), cols);
    EXPECT_EQ(row.topology(), topology::cart);
    EXPECT_EQ(row.rank(), grid.cart_coords<2>()[1]);

    comm const col = cart_sub(grid, {true, false});
    EXPECT_EQ(col.size(), rows);

    comm const all = cart_sub(grid, std::vector<int>{1, 1});
    EXPECT_EQ(all.size(), grid.size());

    // The output-argument form, keeping one dimension. (Keeping none is left out: MPICH and Open MPI differ there.)
    MPI_Comm                  raw = MPI_COMM_NULL;
    std::array<bool, 2> const keep_row{false, true};
    cart_sub(grid, keep_row, raw);
    int size = 0;
    MPI_Comm_size(raw, &size);
    EXPECT_EQ(size, cols);
    MPI_Comm_free(&raw);
}
