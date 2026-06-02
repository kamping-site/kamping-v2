// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <cstddef>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/ialltoallv.hpp"
#include "kamping/v2/request.hpp"
#include "kamping/v2/views.hpp"
#include "kamping/v2/views/auto_recv_v.hpp"

using namespace ::testing;

// Rank i sends (i+1) copies of (rank*10 + j) to rank j.
static std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> build_send(int rank, int size) {
    std::vector<int> counts(static_cast<std::size_t>(size));
    std::vector<int> data;
    for (int j = 0; j < size; ++j) {
        counts[static_cast<std::size_t>(j)] = rank + 1;
        for (int k = 0; k < rank + 1; ++k) {
            data.push_back(rank * 10 + j);
        }
    }
    std::vector<int> displs(static_cast<std::size_t>(size));
    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
    return {data, counts, displs};
}

// High-level overload, borrowed recv buffer — data lands in recv_data after wait().
TEST(V2IalltoallvTest, LvalueRecvBuf) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto [send_data, send_counts, send_displs] = build_send(rank, size);

    std::vector<int> recv_data;
    kamping::v2::ialltoallv(
        send_data | kamping::v2::views::with_counts(send_counts) | kamping::v2::views::with_displs(send_displs),
        recv_data | kamping::v2::views::auto_recv_v
    ).wait();

    std::vector<int> expected;
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < i + 1; ++k) {
            expected.push_back(i * 10 + rank);
        }
    }
    EXPECT_EQ(recv_data, expected);
}

// High-level overload, owned recv buffer — data lives inside the waited result.
TEST(V2IalltoallvTest, OwnedAutoRecvV) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto [send_data, send_counts, send_displs] = build_send(rank, size);

    auto res = kamping::v2::ialltoallv(
        send_data | kamping::v2::views::with_counts(send_counts) | kamping::v2::views::with_displs(send_displs),
        kamping::v2::auto_recv_v<int>()
    ).wait();

    std::vector<int> recv_data(std::ranges::begin(res.recv), std::ranges::end(res.recv));

    std::vector<int> expected;
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < i + 1; ++k) {
            expected.push_back(i * 10 + rank);
        }
    }
    EXPECT_EQ(recv_data, expected);
}

// Degenerate: each rank sends nothing to everyone.
TEST(V2IalltoallvTest, AllEmpty) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data;
    std::vector<int> send_counts(static_cast<std::size_t>(size), 0);
    std::vector<int> send_displs(static_cast<std::size_t>(size), 0);

    std::vector<int> recv_data;
    kamping::v2::ialltoallv(
        send_data | kamping::v2::views::with_counts(send_counts) | kamping::v2::views::with_displs(send_displs),
        recv_data | kamping::v2::views::auto_recv_v
    ).wait();

    EXPECT_TRUE(recv_data.empty());
}

// test() poll eventually completes.
TEST(V2IalltoallvTest, TestPoll) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto [send_data, send_counts, send_displs] = build_send(rank, size);

    std::vector<int> recv_data;
    auto             res = kamping::v2::ialltoallv(
        send_data | kamping::v2::views::with_counts(send_counts) | kamping::v2::views::with_displs(send_displs),
        recv_data | kamping::v2::views::auto_recv_v
    );
    while (!res.test()) {
    }

    std::vector<int> expected;
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < i + 1; ++k) {
            expected.push_back(i * 10 + rank);
        }
    }
    EXPECT_EQ(recv_data, expected);
}

// Low-level overload: caller-owned raw MPI_Request.
TEST(V2IalltoallvTest, ExternalRequest) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto [send_data, send_counts, send_displs] = build_send(rank, size);

    std::vector<int> recv_data;
    MPI_Request      req = MPI_REQUEST_NULL;
    kamping::v2::ialltoallv(
        req,
        send_data | kamping::v2::views::with_counts(send_counts) | kamping::v2::views::with_displs(send_displs),
        recv_data | kamping::v2::views::auto_recv_v
    );
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    std::vector<int> expected;
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < i + 1; ++k) {
            expected.push_back(i * 10 + rank);
        }
    }
    EXPECT_EQ(recv_data, expected);
}

// Low-level overload: caller-owned request_view.
TEST(V2IalltoallvTest, ExternalRequestView) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto [send_data, send_counts, send_displs] = build_send(rank, size);

    std::vector<int>                     recv_data;
    MPI_Request                          req = MPI_REQUEST_NULL;
    mpi::experimental::request_view rv{req};
    kamping::v2::ialltoallv(
        rv,
        send_data | kamping::v2::views::with_counts(send_counts) | kamping::v2::views::with_displs(send_displs),
        recv_data | kamping::v2::views::auto_recv_v
    );
    rv.wait();

    std::vector<int> expected;
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < i + 1; ++k) {
            expected.push_back(i * 10 + rank);
        }
    }
    EXPECT_EQ(recv_data, expected);
}
