// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0
//
// Custom gtest+MPI entry point for the dstl tests that need MPI_THREAD_MULTIPLE. Mirrors KaTestrophe's
// mpi_gtest_main.cpp but initializes MPI with MPI_Init_thread(MPI_THREAD_MULTIPLE) instead of MPI_Init,
// so the thread_multiple_comm exchange can be exercised. The provided level is recorded for skipping.

#include <stdexcept>

#include <gtest/gtest.h>
#include <mpi.h>

#include "gtest-mpi-listener.hpp"
#include "thread_multiple_test_main.hpp"

namespace {
int g_provided_thread_level = MPI_THREAD_SINGLE;
} // namespace

int provided_thread_level() {
    return g_provided_thread_level;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &g_provided_thread_level);

    int init_flag = 0;
    MPI_Initialized(&init_flag);
    if (!init_flag) {
        throw std::runtime_error("MPI was not initialized");
    }

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    ::testing::TestEventListener*  l         = listeners.Release(listeners.default_result_printer());
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));

    return RUN_ALL_TESTS();
}
