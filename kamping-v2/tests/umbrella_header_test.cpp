// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

// Compiles the umbrella header, which catches name clashes between the re-exported core names and the v2 layer.

#include <vector>

#include <gtest/gtest.h>

#include "kamping/v2.hpp"

TEST(UmbrellaHeaderTest, ReexportsBufferContract) {
    std::vector<int> v{1, 2, 3};
    static_assert(kamping::v2::send_buffer<std::vector<int>&>);
    static_assert(kamping::v2::recv_buffer<std::vector<int>&>);
    static_assert(kamping::v2::convertible_to_mpi_handle<MPI_Comm, MPI_Comm>);
    static_assert(kamping::v2::rank<int>);
    EXPECT_EQ(kamping::v2::count(v), 3u);
    EXPECT_EQ(kamping::v2::ptr(v), v.data());
    EXPECT_EQ(kamping::v2::type(v), MPI_INT);
    EXPECT_EQ(kamping::v2::count(v | kamping::v2::views::with_count(2)), 2);
}
