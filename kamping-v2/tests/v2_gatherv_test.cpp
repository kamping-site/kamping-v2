#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/gatherv.hpp"
#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views.hpp"
#include "kamping/v2/views/resize_v_view.hpp"

using namespace ::testing;

class GathervTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }
    int rank_;
    int size_;
};

// Each rank r sends r+1 copies of r: root collects [0, 1,1, 2,2,2, ...].
TEST_F(GathervTest, variable_length_send_auto_recv) {
    std::vector<int> send_data(static_cast<std::size_t>(rank_) + 1, rank_);

    if (rank_ == 0) {
        std::vector<int> recv_data;
        auto [s, r] = kamping::v2::gatherv(
            send_data,
            recv_data | kamping::v2::views::auto_counts() | kamping::v2::views::auto_displs()
                | kamping::v2::views::resize_v
        );
        std::vector<int> expected;
        for (int i = 0; i < size_; ++i) {
            expected.insert(expected.end(), static_cast<std::size_t>(i) + 1, i);
        }
        EXPECT_EQ(recv_data, expected);
    } else {
        kamping::v2::gatherv(send_data, kamping::v2::auto_null_recv_v());
    }
}

// Uniform send count — same result as gather but via gatherv.
TEST_F(GathervTest, uniform_send_count) {
    std::vector<int> send_data = {2 * rank_, 2 * rank_ + 1};

    if (rank_ == 0) {
        std::vector<int> recv_data;
        kamping::v2::gatherv(
            send_data,
            recv_data | kamping::v2::views::auto_counts() | kamping::v2::views::auto_displs()
                | kamping::v2::views::resize_v
        );
        std::vector<int> expected(static_cast<std::size_t>(2 * size_));
        std::iota(expected.begin(), expected.end(), 0);
        EXPECT_EQ(recv_data, expected);
    } else {
        kamping::v2::gatherv(send_data, kamping::v2::auto_null_recv_v());
    }
}

// Non-default root.
TEST_F(GathervTest, non_default_root) {
    int const        root = size_ - 1;
    std::vector<int> send_data(static_cast<std::size_t>(rank_) + 1, rank_);

    if (rank_ == root) {
        std::vector<int> recv_data;
        kamping::v2::gatherv(
            send_data,
            recv_data | kamping::v2::views::auto_counts() | kamping::v2::views::auto_displs()
                | kamping::v2::views::resize_v,
            root
        );
        std::vector<int> expected;
        for (int i = 0; i < size_; ++i) {
            expected.insert(expected.end(), static_cast<std::size_t>(i) + 1, i);
        }
        EXPECT_EQ(recv_data, expected);
    } else {
        kamping::v2::gatherv(send_data, kamping::v2::auto_null_recv_v(), root);
    }
}
