#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/allgatherv.hpp"
#include "kamping/v2/views.hpp"
#include "kamping/v2/views/resize_v_view.hpp"

using namespace ::testing;

// Each rank r sends r+1 copies of r: result is [0, 1,1, 2,2,2, ...].
TEST(V2AllgathervTest, VariableLengthSend) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data(static_cast<std::size_t>(rank) + 1, rank);
    std::vector<int> recv_data;

    kamping::v2::allgatherv(
        send_data,
        recv_data | kamping::v2::views::auto_counts() | kamping::v2::views::auto_displs() | kamping::v2::views::resize_v
    );

    std::vector<int> expected;
    for (int r = 0; r < size; ++r) {
        expected.insert(expected.end(), static_cast<std::size_t>(r) + 1, r);
    }
    EXPECT_EQ(recv_data, expected);
}

// All ranks send exactly one element — same result as allgather.
TEST(V2AllgathervTest, UniformSingleElementSend) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data{rank};
    std::vector<int> recv_data;

    kamping::v2::allgatherv(
        send_data,
        recv_data | kamping::v2::views::auto_counts() | kamping::v2::views::auto_displs() | kamping::v2::views::resize_v
    );

    std::vector<int> expected(static_cast<std::size_t>(size));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(recv_data, expected);
}

// Rank 0 sends nothing; rank r sends r elements of value r.
TEST(V2AllgathervTest, RankZeroSendsEmptyBuffer) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data(static_cast<std::size_t>(rank), rank); // rank 0 → empty
    std::vector<int> recv_data;

    kamping::v2::allgatherv(
        send_data,
        recv_data | kamping::v2::views::auto_counts() | kamping::v2::views::auto_displs() | kamping::v2::views::resize_v
    );

    std::vector<int> expected;
    for (int r = 1; r < size; ++r) {
        expected.insert(expected.end(), static_cast<std::size_t>(r), r);
    }
    EXPECT_EQ(recv_data, expected);
}

// User-provided counts buffer (no auto-resize of counts).
TEST(V2AllgathervTest, UserProvidedCountsBuffer) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data(static_cast<std::size_t>(rank) + 1, rank);
    std::vector<int> recv_data;
    std::vector<int> counts(static_cast<std::size_t>(size)); // pre-sized, no auto-resize

    kamping::v2::allgatherv(
        send_data,
        recv_data | kamping::v2::views::auto_counts(counts) | kamping::v2::views::auto_displs()
            | kamping::v2::views::resize_v
    );

    std::vector<int> expected;
    for (int r = 0; r < size; ++r) {
        expected.insert(expected.end(), static_cast<std::size_t>(r) + 1, r);
    }
    EXPECT_EQ(recv_data, expected);
    // counts should have been filled by infer()
    EXPECT_EQ(counts[static_cast<std::size_t>(rank)], rank + 1);
}

// User-provided counts buffer (no auto-resize of counts).
TEST(V2AllgathervTest, ExplicitCountsAutoDisplsNoResize) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data(static_cast<std::size_t>(rank) + 1, rank);
    std::vector<int> recv_data(static_cast<std::size_t>(size) * (static_cast<std::size_t>(size) + 1) / 2);
    std::vector<int> counts(static_cast<std::size_t>(size)); // pre-sized, no auto-resize
    std::ranges::iota(counts, 1);

    kamping::v2::allgatherv(
        send_data,
        recv_data | kamping::v2::views::with_counts(counts) | kamping::v2::views::auto_displs()
    );

    std::vector<int> expected;
    for (int r = 0; r < size; ++r) {
        expected.insert(expected.end(), static_cast<std::size_t>(r) + 1, r);
    }
    EXPECT_EQ(recv_data, expected);
}
