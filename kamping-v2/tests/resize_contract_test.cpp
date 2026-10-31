// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/buffer.hpp"
#include "kamping/v2/views/resize_view.hpp"
#include "kamping/v2/views/view_interface.hpp"
#include "kamping/v2/views/with_type_view.hpp"

// The resize contract: resize_for_receive(n) keeps the values of [0, min(old size, n)).

namespace views = kamping::v2::views;

namespace {
/// A buffer with its own resize protocol, implementing the contract without value-initializing new elements.
struct custom_buffer {
    std::vector<int> data;
    int              resizes = 0;

    void mpi_resize_for_receive(std::ptrdiff_t n) {
        ++resizes;
        data.resize(static_cast<std::size_t>(n));
    }
    [[nodiscard]] int* mpi_ptr() {
        return data.data();
    }
    [[nodiscard]] std::ptrdiff_t mpi_count() const {
        return static_cast<std::ptrdiff_t>(data.size());
    }
    [[nodiscard]] MPI_Datatype mpi_type() const {
        return MPI_INT;
    }
};
} // namespace

TEST(ResizeContractTest, StandardContainerKeepsPrefixWhenGrowingAndShrinking) {
    std::vector<int> data{1, 2, 3};
    kamping::v2::resize_for_receive(data, 5);
    ASSERT_EQ(data.size(), 5u);
    EXPECT_EQ(data[0], 1);
    EXPECT_EQ(data[1], 2);
    EXPECT_EQ(data[2], 3);

    kamping::v2::resize_for_receive(data, 2);
    EXPECT_EQ(data, (std::vector<int>{1, 2}));
}

TEST(ResizeContractTest, ResizeViewGrowsBehindReceivedData) {
    // Receive 2 elements, then grow by 3 more: the first 2 must survive the second resize.
    std::vector<int> data;
    auto             buf = data | views::resize;

    buf.set_recv_count(2);
    int* first = kamping::v2::ptr(buf);
    first[0]   = 10;
    first[1]   = 11;

    buf.set_recv_count(5);
    int* grown = kamping::v2::ptr(buf);
    EXPECT_EQ(kamping::v2::count(buf), 5);
    ASSERT_EQ(data.size(), 5u);
    EXPECT_EQ(grown[0], 10);
    EXPECT_EQ(grown[1], 11);
}

TEST(ResizeContractTest, ResizeViewOverWrappedBufferKeepsPrefix) {
    // The resize is forwarded through other views (here with_type) via view_interface::mpi_resize_for_receive.
    std::vector<int> data;
    auto             buf = data | views::with_type(MPI_INT) | views::resize;

    buf.set_recv_count(1);
    kamping::v2::ptr(buf)[0] = 42;
    buf.set_recv_count(3);
    kamping::v2::materialize(buf);
    ASSERT_EQ(data.size(), 3u);
    EXPECT_EQ(data[0], 42);
}

TEST(ResizeContractTest, ResizeViewUsesCustomProtocol) {
    static_assert(kamping::v2::has_mpi_resize_for_receive<custom_buffer>);
    custom_buffer storage;
    auto          buf = storage | views::resize;

    buf.set_recv_count(1);
    kamping::v2::ptr(buf)[0] = 7;
    buf.set_recv_count(4);
    kamping::v2::materialize(buf);
    EXPECT_EQ(storage.resizes, 2);
    ASSERT_EQ(storage.data.size(), 4u);
    EXPECT_EQ(storage.data[0], 7);
}
