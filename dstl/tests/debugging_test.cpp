// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dstl/debugging.hpp"
#include "kamping/v2/comm.hpp"

// The tests run without a debugger, so no rank may be selected to wait: they check the rank-list parsing and that
// the non-waiting paths return on every rank.

TEST(DebuggingTest, rank_list_single_ranks) {
    EXPECT_TRUE(dstl::detail::rank_list_contains("3", 3));
    EXPECT_FALSE(dstl::detail::rank_list_contains("3", 2));
    EXPECT_TRUE(dstl::detail::rank_list_contains("0,3", 0));
    EXPECT_TRUE(dstl::detail::rank_list_contains("0,3", 3));
    EXPECT_FALSE(dstl::detail::rank_list_contains("0,3", 1));
}

TEST(DebuggingTest, rank_list_ranges_and_all) {
    EXPECT_TRUE(dstl::detail::rank_list_contains("4-7", 4));
    EXPECT_TRUE(dstl::detail::rank_list_contains("4-7", 7));
    EXPECT_FALSE(dstl::detail::rank_list_contains("4-7", 8));
    EXPECT_TRUE(dstl::detail::rank_list_contains("0,4-7", 0));
    EXPECT_TRUE(dstl::detail::rank_list_contains("all", 12));
}

TEST(DebuggingTest, rank_list_malformed) {
    EXPECT_THROW(dstl::detail::rank_list_contains("", 0), std::invalid_argument);
    EXPECT_THROW(dstl::detail::rank_list_contains("0,", 0), std::invalid_argument);
    EXPECT_THROW(dstl::detail::rank_list_contains("a", 0), std::invalid_argument);
    EXPECT_THROW(dstl::detail::rank_list_contains("1-x", 0), std::invalid_argument);
    EXPECT_THROW(dstl::detail::rank_list_contains("-1", 0), std::invalid_argument);
    EXPECT_THROW(dstl::detail::rank_list_contains("0:3", 0), std::invalid_argument);
}

TEST(DebuggingTest, uses_std_debugging_where_available) {
#if defined(DSTL_HAS_STD_DEBUGGING) && defined(__cpp_lib_debugging)
    EXPECT_EQ(dstl::is_debugger_present(), std::is_debugger_present());
#else
    GTEST_SKIP() << "the standard library does not provide <debugging>";
#endif
}

TEST(DebuggingTest, breakpoint_if_debugging_without_debugger) {
    if (!dstl::is_debugger_present()) {
        dstl::breakpoint_if_debugging(); // must not stop or terminate
    }
    SUCCEED();
}

TEST(DebuggingTest, wait_for_debugger_on_no_rank_is_a_barrier) {
    dstl::wait_for_debugger(std::vector<int>{});
    dstl::wait_for_debugger(std::vector<int>{-1, 1000}, kamping::v2::comm_world());
    dstl::wait_for_debugger(1000);
}

TEST(DebuggingTest, wait_for_debugger_if_requested) {
    char const* const env_var = "DSTL_TEST_WAIT_FOR_DEBUGGER";
    unsetenv(env_var);
    dstl::wait_for_debugger_if_requested(kamping::v2::comm_world(), env_var);

    setenv(env_var, "", 1);
    dstl::wait_for_debugger_if_requested(kamping::v2::comm_world(), env_var);

    setenv(env_var, "1000-2000", 1);
    dstl::wait_for_debugger_if_requested(kamping::v2::comm_world(), env_var);

    setenv(env_var, "not a rank", 1);
    EXPECT_THROW(dstl::wait_for_debugger_if_requested(kamping::v2::comm_world(), env_var), std::invalid_argument);
    unsetenv(env_var);
}
