// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <concepts>
#include <ranges>
#include <span>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/views.hpp"
#include "kamping/v2/views/chunked_view.hpp"
#include "kamping/v2/views/payload.hpp"

namespace views = kamping::v2::views;

namespace {
std::vector<int> const data{0, 1, 2, 3, 4, 5};
std::vector<int> const counts{2, 0, 3, 1};
std::vector<int> const displs{0, 2, 2, 5};
} // namespace

TEST(ChunkedViewTest, ChunksFollowCountsAndDispls) {
    auto chunks = data | views::with_counts(counts) | views::with_displs(displs) | views::chunked;
    ASSERT_EQ(chunks.size(), 4u);
    EXPECT_EQ(std::ranges::size(chunks[0]), 2u);
    EXPECT_EQ(chunks[0][1], 1);
    EXPECT_TRUE(chunks[1].empty());
    EXPECT_EQ(std::vector<int>(chunks[2].begin(), chunks[2].end()), (std::vector<int>{2, 3, 4}));
    EXPECT_EQ(std::vector<int>(chunks[3].begin(), chunks[3].end()), (std::vector<int>{5}));
}

TEST(ChunkedViewTest, ChunksReferToTheBuffer) {
    auto chunks = data | views::with_counts(counts) | views::with_displs(displs) | views::chunked;
    EXPECT_EQ(chunks[2].data(), data.data() + 2);
}

TEST(ChunkedViewTest, WorksWithComputedDisplacements) {
    auto                          chunks = data | views::with_counts(counts) | views::auto_displs() | views::chunked;
    std::vector<std::vector<int>> as_vectors;
    for (auto chunk: chunks) {
        as_vectors.emplace_back(chunk.begin(), chunk.end());
    }
    EXPECT_EQ(as_vectors, (std::vector<std::vector<int>>{{0, 1}, {}, {2, 3, 4}, {5}}));
}

TEST(ChunkedViewTest, IsADenseNestedBufferOfBorrowedMessages) {
    using Chunks = decltype(data | views::with_counts(counts) | views::auto_displs() | views::chunked);
    static_assert(std::ranges::random_access_range<Chunks>);
    static_assert(std::ranges::sized_range<Chunks>);
    static_assert(kamping::v2::nested_send_buffer<Chunks>);
    static_assert(!kamping::v2::sparse_nested_send_buffer<Chunks>);
    static_assert(std::same_as<kamping::v2::nested_message_t<Chunks>, std::span<int const>>);
    static_assert(kamping::v2::buffer_message<std::span<int const>>);
    static_assert(kamping::v2::borrowed_buffer<std::span<int const>>);
}

TEST(ChunkedViewTest, ValueTypeIsTheBufferType) {
    auto plain = data | views::with_counts(counts) | views::auto_displs() | views::chunked;
    EXPECT_EQ(kamping::v2::value_type(plain), MPI_INT);
    auto typed =
        data | views::with_type(MPI_UNSIGNED) | views::with_counts(counts) | views::auto_displs() | views::chunked;
    EXPECT_EQ(kamping::v2::value_type(typed), MPI_UNSIGNED);
}

TEST(ChunkedViewTest, FlattenRoundTrips) {
    // flatten_v(chunked(x)) gives back the data and counts.
    auto        flat = data | views::with_counts(counts) | views::auto_displs() | views::chunked | views::flatten_v();
    auto const* p    = static_cast<int const*>(mpi::experimental::ptr(flat));
    EXPECT_EQ(std::vector<int>(p, p + mpi::experimental::count(flat)), data);
    auto const& flat_counts = mpi::experimental::counts(flat);
    EXPECT_EQ(std::vector<int>(std::ranges::begin(flat_counts), std::ranges::end(flat_counts)), counts);
}
