// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <type_traits>

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/comm.hpp"

using namespace mpi::experimental;

// ── Predefined communicators ──────────────────────────────────────────────────

static_assert(std::is_same_v<decltype(comm_world()), comm_view>);
static_assert(std::is_same_v<decltype(comm_self()), comm_view>);

TEST(CommTest, CommWorldIsMpiCommWorld) {
    comm_view world = comm_world();
    EXPECT_EQ(world.mpi_handle(), MPI_COMM_WORLD);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    EXPECT_EQ(world.rank(), rank);
    EXPECT_EQ(world.size(), size);
}

TEST(CommTest, CommSelfIsMpiCommSelf) {
    comm_view self = comm_self();
    EXPECT_EQ(self.mpi_handle(), MPI_COMM_SELF);
    EXPECT_EQ(self.rank(), 0);
    EXPECT_EQ(self.size(), 1);
}

TEST(CommTest, CommWorldIsUsableAsHandle) {
    EXPECT_EQ(mpi::experimental::handle(comm_world()), MPI_COMM_WORLD);
    comm duplicate = comm_world().dup();
    EXPECT_EQ(duplicate.size(), comm_world().size());
}
