#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/scatter.hpp"
#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views/resize_view.hpp"

using namespace ::testing;

class ScatterTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }
    int rank_;
    int size_;
};

TEST_F(ScatterTest, scatter_simple) {
    std::vector<int> recv_data(2);

    if (rank_ == 0) {
        std::vector<int> send_data(static_cast<std::size_t>(2 * size_));
        std::iota(send_data.begin(), send_data.end(), 0);
        kamping::v2::scatter(send_data, recv_data);
    } else {
        kamping::v2::scatter(recv_data);
    }

    EXPECT_EQ(recv_data[0], 2 * rank_);
    EXPECT_EQ(recv_data[1], 2 * rank_ + 1);
}

TEST_F(ScatterTest, scatter_resize) {
    std::vector<int> recv_data;

    if (rank_ == 0) {
        std::vector<int> send_data(static_cast<std::size_t>(2 * size_));
        std::iota(send_data.begin(), send_data.end(), 0);
        kamping::v2::scatter(send_data, recv_data | kamping::v2::views::resize);
    } else {
        kamping::v2::scatter(recv_data | kamping::v2::views::resize);
    }

    EXPECT_EQ(recv_data[0], 2 * rank_);
    EXPECT_EQ(recv_data[1], 2 * rank_ + 1);
}

TEST_F(ScatterTest, scatter_inplace_on_root) {
    std::vector<int> send_data(static_cast<std::size_t>(2 * size_));
    std::iota(send_data.begin(), send_data.end(), 0);

    if (rank_ == 0) {
        kamping::v2::scatter(send_data, kamping::v2::inplace);
        // root's portion is already at offset 0 in send_data, untouched
        EXPECT_EQ(send_data[0], 0);
        EXPECT_EQ(send_data[1], 1);
    } else {
        std::vector<int> recv_data(2);
        kamping::v2::scatter(recv_data);
        EXPECT_EQ(recv_data[0], 2 * rank_);
        EXPECT_EQ(recv_data[1], 2 * rank_ + 1);
    }
}

TEST_F(ScatterTest, scatter_non_default_root) {
    int const        root = size_ - 1;
    std::vector<int> recv_data(2);

    if (rank_ == root) {
        std::vector<int> send_data(static_cast<std::size_t>(2 * size_));
        std::iota(send_data.begin(), send_data.end(), 0);
        kamping::v2::scatter(send_data, recv_data, root);
    } else {
        kamping::v2::scatter(recv_data, root);
    }

    EXPECT_EQ(recv_data[0], 2 * rank_);
    EXPECT_EQ(recv_data[1], 2 * rank_ + 1);
}
