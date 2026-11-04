// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <utility>

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/comm.hpp"
#include "mpi/win.hpp"

TEST(WinTest, DefaultIsNull) {
    mpi::experimental::win w;
    EXPECT_EQ(mpi::experimental::handle(w), MPI_WIN_NULL);
}

TEST(WinTest, AllocateExposesLocalMemory) {
    auto [w, local] = mpi::experimental::win_allocate<double>(4, MPI_COMM_WORLD);
    EXPECT_NE(mpi::experimental::handle(w), MPI_WIN_NULL);
    EXPECT_EQ(local.size(), 4u);
    local[3] = 1.5; // writable
    EXPECT_EQ(local[3], 1.5);
}

TEST(WinTest, AccumulateWithFences) {
    // Every rank adds 1 to rank 0's counter; afterwards it holds the communicator size.
    mpi::experimental::comm_view world{MPI_COMM_WORLD};
    auto [w, local] = mpi::experimental::win_allocate<int>(1, MPI_COMM_WORLD);
    local[0]        = 0;
    w.fence(MPI_MODE_NOPRECEDE);
    int const one = 1;
    MPI_Accumulate(&one, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM, mpi::experimental::handle(w));
    w.fence(MPI_MODE_NOSUCCEED);
    if (world.rank() == 0) {
        EXPECT_EQ(local[0], world.size());
    }
}

TEST(WinTest, LevelOneAllocateIntoOutputHandle) {
    mpi::experimental::win w;
    int*                   base = nullptr;
    mpi::experimental::win_allocate(2, MPI_COMM_WORLD, base, w);
    EXPECT_NE(mpi::experimental::handle(w), MPI_WIN_NULL);
    EXPECT_NE(base, nullptr);
    mpi::experimental::fence(w); // free function, same as w.fence()
}

TEST(WinTest, ViewAndGroup) {
    mpi::experimental::comm_view world{MPI_COMM_WORLD};
    auto [w, local]                  = mpi::experimental::win_allocate<int>(1, MPI_COMM_WORLD);
    mpi::experimental::win_view view = w;
    EXPECT_EQ(mpi::experimental::handle(view), mpi::experimental::handle(w));
    EXPECT_EQ(view.group().size(), world.size());
    view.fence();
}

TEST(WinTest, MoveTransfersOwnership) {
    auto [w, local]              = mpi::experimental::win_allocate<int>(1, MPI_COMM_WORLD);
    MPI_Win const          raw   = mpi::experimental::handle(w);
    mpi::experimental::win moved = std::move(w);
    EXPECT_EQ(mpi::experimental::handle(moved), raw);
    EXPECT_EQ(mpi::experimental::handle(w), MPI_WIN_NULL);
}

TEST(WinTest, FreeAndRelease) {
    auto [w, local] = mpi::experimental::win_allocate<int>(1, MPI_COMM_WORLD);
    w.free();
    EXPECT_EQ(mpi::experimental::handle(w), MPI_WIN_NULL);
    w.free(); // no-op on a null window

    auto [w2, local2] = mpi::experimental::win_allocate<int>(1, MPI_COMM_WORLD);
    MPI_Win raw       = w2.release();
    EXPECT_EQ(mpi::experimental::handle(w2), MPI_WIN_NULL);
    mpi::experimental::win adopted = mpi::experimental::win::from_native(raw);
    EXPECT_EQ(mpi::experimental::handle(adopted), raw);
}
