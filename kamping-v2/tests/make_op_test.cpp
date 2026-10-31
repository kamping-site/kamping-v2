// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <algorithm>
#include <functional>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/allreduce.hpp"
#include "kamping/v2/collectives/reduce.hpp"
#include "kamping/v2/ops.hpp"

using namespace ::kamping;

class MakeOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    int rank_;
    int size_;
};

// A default-constructible (stateless) binary functor.
struct MaxFunctor {
    int operator()(int const& a, int const& b) const {
        return std::max(a, b);
    }
};

// A raw MPI reduction callback summing ints element-wise.
static void int_sum_callback(void* invec, void* inoutvec, int* len, MPI_Datatype* /*datatype*/) {
    int* in    = static_cast<int*>(invec);
    int* inout = static_cast<int*>(inoutvec);
    for (int i = 0; i < *len; ++i) {
        inout[i] += in[i];
    }
}

TEST_F(MakeOpTest, functor_from_captureless_lambda) {
    std::vector<int> send_data = {rank_, rank_ + 1};
    std::vector<int> recv_data(2);

    auto op = kamping::v2::make_op<int>([](int a, int b) { return a + b; });
    kamping::v2::reduce(send_data, recv_data, op, 0);

    if (rank_ == 0) {
        int const sum_ranks = size_ * (size_ - 1) / 2;
        EXPECT_EQ(recv_data[0], sum_ranks);
        EXPECT_EQ(recv_data[1], sum_ranks + size_);
    }
}

TEST_F(MakeOpTest, functor_from_struct_with_commutative_tag) {
    std::vector<int> send_data = {rank_};
    std::vector<int> recv_data(1);

    auto op = kamping::v2::make_op<int>(MaxFunctor{}, kamping::v2::commutative);
    kamping::v2::allreduce(send_data, recv_data, op);

    EXPECT_EQ(recv_data[0], size_ - 1);
}

TEST_F(MakeOpTest, functor_default_is_non_commutative) {
    // Non-commutative is only a correctness-preserving hint to MPI; the numeric result of a
    // genuinely commutative op is unaffected. We just check the default overload compiles + runs.
    std::vector<int> send_data = {rank_ + 1};
    std::vector<int> recv_data(1);

    auto op = kamping::v2::make_op<int>(std::plus<int>{});
    kamping::v2::allreduce(send_data, recv_data, op);

    int const expected = size_ * (size_ + 1) / 2;
    EXPECT_EQ(recv_data[0], expected);
}

TEST_F(MakeOpTest, raw_mpi_callback) {
    std::vector<int> send_data = {rank_, 2 * rank_};
    std::vector<int> recv_data(2);

    auto op = kamping::v2::make_op(&int_sum_callback, kamping::v2::commutative);
    kamping::v2::allreduce(send_data, recv_data, op);

    int const sum_ranks = size_ * (size_ - 1) / 2;
    EXPECT_EQ(recv_data[0], sum_ranks);
    EXPECT_EQ(recv_data[1], 2 * sum_ranks);
}
