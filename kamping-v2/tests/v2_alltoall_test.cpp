#include <cstddef>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/alltoall.hpp"
#include "kamping/v2/views.hpp"

using namespace ::testing;

// Each rank sends its rank to every other rank; every rank receives all ranks.
TEST(V2AlltoallTest, SendRankToAll) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // send_buf[j] = rank for all j: this rank sends its rank value to every destination.
    std::vector<int> send_data(static_cast<std::size_t>(size), rank);
    std::vector<int> recv_data(static_cast<std::size_t>(size));

    kamping::v2::alltoall(send_data, recv_data);

    // recv_data[i] = i: received the rank of sender i from each sender.
    std::vector<int> expected(static_cast<std::size_t>(size));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(recv_data, expected);
}

// Deferred recv buffer: resize via infer().
TEST(V2AlltoallTest, DeferredRecvBuffer) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data(static_cast<std::size_t>(size), rank);
    std::vector<int> recv_data;

    kamping::v2::alltoall(send_data, recv_data | kamping::v2::views::resize);

    std::vector<int> expected(static_cast<std::size_t>(size));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(recv_data, expected);
}

// Transpose: send_buf[j] = rank * size + j; each rank ends up with a column.
TEST(V2AlltoallTest, Transpose) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data(static_cast<std::size_t>(size));
    for (int j = 0; j < size; ++j) {
        send_data[static_cast<std::size_t>(j)] = rank * size + j;
    }
    std::vector<int> recv_data(static_cast<std::size_t>(size));

    kamping::v2::alltoall(send_data, recv_data);

    // recv_data[i] = i * size + rank: the rank-th column of the matrix.
    std::vector<int> expected(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i) {
        expected[static_cast<std::size_t>(i)] = i * size + rank;
    }
    EXPECT_EQ(recv_data, expected);
}
