// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <cstddef>

#include <gtest/gtest.h>
#include <kamping/types/reduce_ops.hpp>
#include <mpi.h>

#include "dstl/thread_multiple_comm.hpp"
#include "thread_multiple_test_main.hpp"
#include "kamping/v2/collectives/allreduce.hpp"
#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "mpi/comm.hpp"

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

// The number of per-thread comms thread_multiple_comm allocates: the max of omp_get_max_threads()
// over all ranks (see thread_multiple_comm.hpp — the count must be agreed so the collective dups/exchanges match).
std::size_t expected_thread_comm_count() {
    int t = dstl::flat_max_threads();
    kamping::v2::allreduce(kamping::v2::inplace, kamping::v2::views::ref_single(t), kamping::ops::max<>{}, comm_view{MPI_COMM_WORLD});
    return static_cast<std::size_t>(t);
}

} // namespace

// thread_multiple_comm exposes the global rank/size and allocates exactly flat_max_threads() duplicated
// communicators, each the same size as the global communicator.
TEST(ThreadMultipleCommTest, AllocatesThreadComms) {
    if (provided_thread_level() < MPI_THREAD_MULTIPLE) {
        GTEST_SKIP() << "runtime does not provide MPI_THREAD_MULTIPLE";
    }
    dstl::thread_multiple_comm fc{comm_view{MPI_COMM_WORLD}};
    EXPECT_EQ(fc.rank(), world_rank());
    EXPECT_EQ(fc.size(), world_size());
    auto comms = fc.thread_comms();
    EXPECT_EQ(comms.size(), expected_thread_comm_count());
    for (auto const& c: comms) {
        EXPECT_EQ(c.size(), world_size());
        EXPECT_NE(c.mpi_handle(), MPI_COMM_WORLD); // a genuine duplicate, not an alias
    }
}

// Repeated construct/destroy must not leak communicators (the move-only mpi::comm frees its dups).
TEST(ThreadMultipleCommTest, ThreadMultipleFreesThreadComms) {
    if (provided_thread_level() < MPI_THREAD_MULTIPLE) {
        GTEST_SKIP() << "runtime does not provide MPI_THREAD_MULTIPLE";
    }
    auto const expected = expected_thread_comm_count();
    for (int i = 0; i < 50; ++i) {
        dstl::thread_multiple_comm fc{comm_view{MPI_COMM_WORLD}};
        EXPECT_EQ(fc.thread_comms().size(), expected);
    }
    SUCCEED();
}
