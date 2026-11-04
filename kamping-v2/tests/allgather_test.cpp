// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <cstddef>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/allgather.hpp"
#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views.hpp"

namespace {
/// [0, 0, 1, 1, ..., size-1, size-1] for `per_rank` == 2.
std::vector<int> expected_ranks(int size, int per_rank) {
    std::vector<int> expected;
    for (int r = 0; r < size; ++r) {
        expected.insert(expected.end(), static_cast<std::size_t>(per_rank), r);
    }
    return expected;
}
} // namespace

// Pre-sized receive buffer: each rank contributes two copies of its rank.
TEST(V2AllgatherTest, PresizedRecvBuffer) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data(2, rank);
    std::vector<int> recv_data(static_cast<std::size_t>(2 * size));
    kamping::v2::allgather(send_data, recv_data, MPI_COMM_WORLD);

    EXPECT_EQ(recv_data, expected_ranks(size, 2));
}

// Deferred receive buffer: infer() resizes it to comm_size * count(sbuf), keeping no stale elements.
TEST(V2AllgatherTest, DeferredRecvBuffer) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data(3, rank);
    std::vector<int> recv_data(1, -1);
    kamping::v2::allgather(send_data, recv_data | kamping::v2::views::resize, MPI_COMM_WORLD);

    EXPECT_EQ(recv_data, expected_ranks(size, 3));
}

// Owned buffers are moved into and back out of the result; structured bindings give access to both.
TEST(V2AllgatherTest, OwnedBuffersInResult) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto [sent, received] =
        kamping::v2::allgather(std::vector<int>{rank}, kamping::v2::auto_recv<int>(), MPI_COMM_WORLD);

    EXPECT_EQ(sent, std::vector<int>{rank});
    EXPECT_EQ(std::move(received).underlying(), expected_ranks(size, 1));
}

// In place: each rank's contribution is already at its position in the receive buffer.
TEST(V2AllgatherTest, InPlace) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> data(static_cast<std::size_t>(size), -1);
    data[static_cast<std::size_t>(rank)] = rank;
    kamping::v2::allgather(kamping::v2::inplace, data, MPI_COMM_WORLD);

    EXPECT_EQ(data, expected_ranks(size, 1));
}
