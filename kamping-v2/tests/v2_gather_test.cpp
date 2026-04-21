#include "gmock/gmock.h"
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/gather.hpp"
#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views/resize_view.hpp"

class GatherTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    int rank_;
    int size_;
};

TEST_F(GatherTest, gather_simple) {
    std::vector<int> send_data = {2 * rank_, 2 * rank_ + 1};
    std::vector<int> recv_data(static_cast<std::size_t>(2 * size_));

    kamping::v2::gather(send_data, recv_data);
    if (rank_ == 0) {
        std::vector<int> expected(static_cast<std::size_t>(2 * size_));
        std::ranges::iota(expected, 0);
        EXPECT_THAT(recv_data, testing::ElementsAreArray(expected));
    }
}

TEST_F(GatherTest, gather_recv_buf_only_on_root) {
    std::vector<int> send_data = {2 * rank_, 2 * rank_ + 1};

    if (rank_ == 0) {
        std::vector<int> recv_data(static_cast<std::size_t>(2 * size_));
        kamping::v2::gather(send_data, recv_data);
        std::vector<int> expected(static_cast<std::size_t>(2 * size_));
        std::ranges::iota(expected, 0);
        EXPECT_THAT(recv_data, testing::ElementsAreArray(expected));
    } else {
        kamping::v2::gather(send_data);
    }
}

TEST_F(GatherTest, gather_resize) {
    std::vector<int> send_data = {2 * rank_, 2 * rank_ + 1};

    if (rank_ == 0) {
        std::vector<int> recv_data(static_cast<std::size_t>(2 * size_));
        kamping::v2::gather(send_data, recv_data | kamping::v2::views::resize);
        std::vector<int> expected(static_cast<std::size_t>(2 * size_));
        std::ranges::iota(expected, 0);
        EXPECT_THAT(recv_data, testing::ElementsAreArray(expected));
    } else {
        kamping::v2::gather(send_data);
    }
}
