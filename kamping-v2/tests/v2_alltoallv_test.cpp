#include <cstddef>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/alltoallv.hpp"
#include "kamping/v2/views.hpp"
#include "kamping/v2/views/auto_recv_v.hpp"

using namespace ::testing;

// Helper: build send buffer + counts + exclusive-scan displs where rank i sends (i+1) copies of
// (rank*10 + j) to rank j.
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

// Lvalue recv buffer: recv_data is borrowed and mutated in place.
TEST(V2AlltoallvTest, LvalueRecvBuf) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto [send_data, send_counts, send_displs] = build_send(rank, size);

    std::vector<int> recv_data;
    kamping::v2::alltoallv(
        send_data | kamping::v2::views::with_counts(send_counts) | kamping::v2::views::with_displs(send_displs),
        recv_data | kamping::v2::views::auto_recv_v
    );

    std::vector<int> expected;
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < i + 1; ++k) {
            expected.push_back(i * 10 + rank);
        }
    }
    EXPECT_EQ(recv_data, expected);
}

// Owned recv buffer via v2::auto_recv_v<T>(): data lives inside the result.
TEST(V2AlltoallvTest, OwnedAutoRecvV) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto [send_data, send_counts, send_displs] = build_send(rank, size);

    auto res = kamping::v2::alltoallv(
        send_data | kamping::v2::views::with_counts(send_counts) | kamping::v2::views::with_displs(send_displs),
        kamping::v2::auto_recv_v<int>()
    );

    std::vector<int> recv_data(std::ranges::begin(res.recv), std::ranges::end(res.recv));

    std::vector<int> expected;
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < i + 1; ++k) {
            expected.push_back(i * 10 + rank);
        }
    }
    EXPECT_EQ(recv_data, expected);
}

// Each rank sends exactly one element to each rank — same result as alltoall.
TEST(V2AlltoallvTest, UniformSingleElement) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data(static_cast<std::size_t>(size), rank);
    std::vector<int> send_counts(static_cast<std::size_t>(size), 1);
    std::vector<int> send_displs(static_cast<std::size_t>(size));
    std::iota(send_displs.begin(), send_displs.end(), 0);

    std::vector<int> recv_data;
    kamping::v2::alltoallv(
        send_data | kamping::v2::views::with_counts(send_counts) | kamping::v2::views::with_displs(send_displs),
        recv_data | kamping::v2::views::auto_recv_v
    );

    std::vector<int> expected(static_cast<std::size_t>(size));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(recv_data, expected);
}

// Degenerate: each rank sends nothing to everyone.
TEST(V2AlltoallvTest, AllEmpty) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data;
    std::vector<int> send_counts(static_cast<std::size_t>(size), 0);
    std::vector<int> send_displs(static_cast<std::size_t>(size), 0);

    std::vector<int> recv_data;
    kamping::v2::alltoallv(
        send_data | kamping::v2::views::with_counts(send_counts) | kamping::v2::views::with_displs(send_displs),
        recv_data | kamping::v2::views::auto_recv_v
    );

    EXPECT_TRUE(recv_data.empty());
}
