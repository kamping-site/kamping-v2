// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/iallgather.hpp"
#include "kamping/v2/request.hpp"
#include "kamping/v2/views.hpp"

using namespace ::testing;

// Each rank contributes its rank value; after wait() the pre-sized recv buffer holds [0, 1, ..., size-1].
TEST(V2IallgatherTest, PreSizedRecvBuffer) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data{rank};
    std::vector<int> recv_data(static_cast<std::size_t>(size));

    kamping::v2::iallgather(send_data, recv_data).wait();

    std::vector<int> expected(static_cast<std::size_t>(size));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(recv_data, expected);
}

// Deferred recv buffer: infer() sizes it to comm_size * send_count before MPI_Iallgather.
TEST(V2IallgatherTest, DeferredRecvBuffer) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data{rank};
    std::vector<int> recv_data;

    kamping::v2::iallgather(send_data, recv_data | kamping::v2::views::resize).wait();

    std::vector<int> expected(static_cast<std::size_t>(size));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(recv_data, expected);
}

// Multiple elements per rank, deferred recv buffer.
TEST(V2IallgatherTest, MultipleElementsPerRank) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data(3, rank);
    std::vector<int> recv_data;

    kamping::v2::iallgather(send_data, recv_data | kamping::v2::views::resize).wait();

    std::vector<int> expected;
    for (int r = 0; r < size; ++r) {
        expected.insert(expected.end(), 3, r);
    }
    EXPECT_EQ(recv_data, expected);
}

// Low-level overload: caller supplies an external request and drives completion manually.
TEST(V2IallgatherTest, ExternalRequest) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data{rank};
    std::vector<int> recv_data(static_cast<std::size_t>(size));

    MPI_Request req = MPI_REQUEST_NULL;
    kamping::v2::iallgather(req, send_data, recv_data);
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    std::vector<int> expected(static_cast<std::size_t>(size));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(recv_data, expected);
}

// External request via request_view wrapper.
TEST(V2IallgatherTest, ExternalRequestView) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data{rank};
    std::vector<int> recv_data(static_cast<std::size_t>(size));

    MPI_Request                          req = MPI_REQUEST_NULL;
    mpi::experimental::request_view rv{req};
    kamping::v2::iallgather(rv, send_data, recv_data);
    rv.wait();

    std::vector<int> expected(static_cast<std::size_t>(size));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(recv_data, expected);
}

// Verify that test() eventually returns a completed result.
TEST(V2IallgatherTest, TestPoll) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data{rank};
    std::vector<int> recv_data(static_cast<std::size_t>(size));

    auto res = kamping::v2::iallgather(send_data, recv_data);
    while (!res.test()) {
    }

    std::vector<int> expected(static_cast<std::size_t>(size));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(recv_data, expected);
}
