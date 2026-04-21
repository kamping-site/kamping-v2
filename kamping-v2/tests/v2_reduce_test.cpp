#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/reduce.hpp"
#include "kamping/v2/views/resize_view.hpp"

using namespace ::kamping;

class ReduceTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    int rank_;
    int size_;
};

TEST_F(ReduceTest, reduce_with_mpi_sum) {
    std::vector<int> send_data = {rank_, rank_ + 1};
    std::vector<int> recv_data(2);

    auto [s, r] = kamping::v2::reduce(send_data, recv_data, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank_ == 0) {
        // Expected: sum of all ranks and sum of (rank + 1)
        int expected_sum_ranks       = size_ * (size_ - 1) / 2;
        int expected_sum_rank_plus_1 = size_ * (size_ - 1) / 2 + size_;
        EXPECT_EQ(r[0], expected_sum_ranks);
        EXPECT_EQ(r[1], expected_sum_rank_plus_1);
    }
}

TEST_F(ReduceTest, reduce_with_builtin_operation) {
    std::vector<double> send_data = {1.0 * rank_, 2.0 * rank_};
    std::vector<double> recv_data(2);

    auto [s, r] = kamping::v2::reduce(send_data, recv_data, MPI_MAX, 0);

    if (rank_ == 0) {
        EXPECT_EQ(r[0], 1.0 * (size_ - 1));
        EXPECT_EQ(r[1], 2.0 * (size_ - 1));
    }
}

TEST_F(ReduceTest, reduce_non_root_rank_unchanged) {
    std::vector<int> send_data = {rank_};
    std::vector<int> recv_data = {999}; // Non-root recv buffer should not change

    kamping::v2::reduce(send_data, recv_data, MPI_SUM, 0);

    if (rank_ != 0) {
        EXPECT_EQ(recv_data[0], 999); // Should not be modified on non-root
    }
}

TEST_F(ReduceTest, reduce_with_std_plus) {
    std::vector<int> send_data = {rank_, rank_ + 10};
    std::vector<int> recv_data(2);

    auto [s, r] = kamping::v2::reduce(send_data, recv_data, std::plus<>{}, 0);

    if (rank_ == 0) {
        int expected_sum_ranks        = size_ * (size_ - 1) / 2;
        int expected_sum_rank_plus_10 = size_ * (size_ - 1) / 2 + 10 * size_;
        EXPECT_EQ(r[0], expected_sum_ranks);
        EXPECT_EQ(r[1], expected_sum_rank_plus_10);
    }
}

TEST_F(ReduceTest, reduce_with_default_arguments) {
    std::vector<int> send_data = {rank_, rank_ + 10};
    std::vector<int> recv_data(2);

    auto [s, r] = kamping::v2::reduce(send_data, recv_data);

    if (rank_ == 0) {
        int expected_sum_ranks        = size_ * (size_ - 1) / 2;
        int expected_sum_rank_plus_10 = size_ * (size_ - 1) / 2 + 10 * size_;
        EXPECT_EQ(r[0], expected_sum_ranks);
        EXPECT_EQ(r[1], expected_sum_rank_plus_10);
    }
}

TEST_F(ReduceTest, reduce_non_root_shorthand_and_inplace_on_root) {
    std::vector<int> data = {rank_, rank_ + 1};

    if (rank_ != 0) {
        // Non-root: one-argument form (implicitly passes null_buf for recv)
        kamping::v2::reduce(data, MPI_SUM, 0);
    } else {
        // Root: inplace form
        kamping::v2::reduce(kamping::v2::inplace, data, MPI_SUM, 0);

        int expected_sum_ranks       = size_ * (size_ - 1) / 2;
        int expected_sum_rank_plus_1 = size_ * (size_ - 1) / 2 + size_;
        EXPECT_EQ(data[0], expected_sum_ranks);
        EXPECT_EQ(data[1], expected_sum_rank_plus_1);
    }
}

TEST_F(ReduceTest, reduce_non_root_shorthand_and_inplace_on_root_resize_not_triggered) {
    std::vector<int> data = {rank_, rank_ + 1};

    if (rank_ != 0) {
        // Non-root: one-argument form (implicitly passes null_buf for recv)
        kamping::v2::reduce(data, MPI_SUM, 0);
    } else {
        // Root: inplace form, resizing should not be triggered and data.size() should be used.
        kamping::v2::reduce(kamping::v2::inplace, data | kamping::v2::views::resize, MPI_SUM, 0);

        int expected_sum_ranks       = size_ * (size_ - 1) / 2;
        int expected_sum_rank_plus_1 = size_ * (size_ - 1) / 2 + size_;
        EXPECT_EQ(data[0], expected_sum_ranks);
        EXPECT_EQ(data[1], expected_sum_rank_plus_1);
    }
}
