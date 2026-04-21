#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/scatterv.hpp"
#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views.hpp"
#include "kamping/v2/views/resize_view.hpp"

using namespace ::testing;

class ScattervTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }
    int rank_;
    int size_;
};

// Root sends rank r exactly r+1 elements of value r.
TEST_F(ScattervTest, variable_length_recv_auto_resize) {
    std::vector<int> recv_data;

    if (rank_ == 0) {
        std::vector<int> send_data;
        std::vector<int> counts(static_cast<std::size_t>(size_));
        for (int r = 0; r < size_; ++r) {
            counts[static_cast<std::size_t>(r)] = r + 1;
            send_data.insert(send_data.end(), static_cast<std::size_t>(r) + 1, r);
        }
        kamping::v2::scatterv(
            send_data | kamping::v2::views::with_counts(counts) | kamping::v2::views::auto_displs(),
            recv_data | kamping::v2::views::resize
        );
    } else {
        kamping::v2::scatterv(kamping::v2::null_buf_v, recv_data | kamping::v2::views::resize);
    }

    EXPECT_EQ(static_cast<int>(recv_data.size()), rank_ + 1);
    EXPECT_THAT(recv_data, Each(rank_));
}

TEST_F(ScattervTest, variable_length_recv_presized) {
    std::vector<int> recv_data(static_cast<std::size_t>(rank_) + 1);

    if (rank_ == 0) {
        std::vector<int> send_data;
        std::vector<int> counts(static_cast<std::size_t>(size_));
        for (int r = 0; r < size_; ++r) {
            counts[static_cast<std::size_t>(r)] = r + 1;
            send_data.insert(send_data.end(), static_cast<std::size_t>(r) + 1, r);
        }
        kamping::v2::scatterv(
            send_data | kamping::v2::views::with_counts(counts) | kamping::v2::views::auto_displs(),
            recv_data
        );
    } else {
        kamping::v2::scatterv(kamping::v2::null_buf_v, recv_data);
    }

    EXPECT_EQ(static_cast<int>(recv_data.size()), rank_ + 1);
    EXPECT_THAT(recv_data, Each(rank_));
}

TEST_F(ScattervTest, non_default_root) {
    int const        root = size_ - 1;
    std::vector<int> recv_data;

    if (rank_ == root) {
        std::vector<int> send_data;
        std::vector<int> counts(static_cast<std::size_t>(size_));
        for (int r = 0; r < size_; ++r) {
            counts[static_cast<std::size_t>(r)] = r + 1;
            send_data.insert(send_data.end(), static_cast<std::size_t>(r) + 1, r);
        }
        kamping::v2::scatterv(
            send_data | kamping::v2::views::with_counts(counts) | kamping::v2::views::auto_displs(),
            recv_data | kamping::v2::views::resize,
            root
        );
    } else {
        kamping::v2::scatterv(kamping::v2::null_buf_v, recv_data | kamping::v2::views::resize, root);
    }

    EXPECT_EQ(static_cast<int>(recv_data.size()), rank_ + 1);
    EXPECT_THAT(recv_data, Each(rank_));
}
