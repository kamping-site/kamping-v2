// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <gtest/gtest.h>

#include "kamping/v2/buffer.hpp"
#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views/with_count_view.hpp"
#include "kamping/v2/views/with_type_view.hpp"

using namespace kamping;

// ── inplace_t ─────────────────────────────────────────────────────────────────

static_assert(kamping::v2::send_buffer<v2::inplace_t>);
static_assert(kamping::v2::recv_buffer<v2::inplace_t>);

TEST(SentinelsTest, InplaceData) {
    EXPECT_EQ(kamping::v2::ptr(v2::inplace), MPI_IN_PLACE);
    EXPECT_EQ(kamping::v2::count(v2::inplace), 0);
    EXPECT_EQ(kamping::v2::type(v2::inplace), MPI_DATATYPE_NULL);
}

// ── null_buf_t ────────────────────────────────────────────────────────────────

static_assert(kamping::v2::send_buffer<v2::null_buf_t>);
static_assert(kamping::v2::recv_buffer<v2::null_buf_t>);

TEST(SentinelsTest, NullBufData) {
    EXPECT_EQ(kamping::v2::ptr(v2::null_buf), nullptr);
    EXPECT_EQ(kamping::v2::count(v2::null_buf), 0);
    EXPECT_EQ(kamping::v2::type(v2::null_buf), MPI_BYTE);
}

// ── bottom_t ──────────────────────────────────────────────────────────────────

// bottom_t alone is not a data_buffer (no size or type)
static_assert(!kamping::v2::data_buffer<v2::bottom_t>);
static_assert(kamping::v2::has_mpi_ptr<v2::bottom_t>);
static_assert(!kamping::v2::has_mpi_count<v2::bottom_t>);
static_assert(!kamping::v2::has_mpi_type<v2::bottom_t>);

TEST(SentinelsTest, BottomData) {
    EXPECT_EQ(kamping::v2::ptr(v2::bottom), MPI_BOTTOM);
}

// Composed with with_type + with_size it becomes a full send_buffer
TEST(SentinelsTest, BottomComposed) {
    auto buf = v2::bottom | v2::views::with_type(MPI_BYTE) | v2::views::with_count(4);
    static_assert(kamping::v2::send_buffer<decltype(buf)>);
    EXPECT_EQ(kamping::v2::ptr(buf), MPI_BOTTOM);
    EXPECT_EQ(kamping::v2::type(buf), MPI_BYTE);
    EXPECT_EQ(kamping::v2::count(buf), 4);
}
