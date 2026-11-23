// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <thread>

#include <gtest/gtest.h>

#include "mpi/time.hpp"

TEST(TimeTest, TimeWorks) {
    using namespace std::chrono_literals;
    auto start = mpi::experimental::clock::now();
    std::this_thread::sleep_for(3s);
    auto end = mpi::experimental::clock::now();
    EXPECT_GE(end - start, 3s);
    EXPECT_GE(mpi::experimental::clock::resolution(), 0s);
}
