#include <gtest/gtest.h>

#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views/with_size_view.hpp"
#include "kamping/v2/views/with_type_view.hpp"

using namespace kamping;

// ── inplace_t ─────────────────────────────────────────────────────────────────

static_assert(mpi::experimental::send_buffer<v2::inplace_t>);
static_assert(mpi::experimental::recv_buffer<v2::inplace_t>);

TEST(SentinelsTest, InplaceData) {
    EXPECT_EQ(mpi::experimental::ptr(v2::inplace), MPI_IN_PLACE);
    EXPECT_EQ(mpi::experimental::count(v2::inplace), 0);
    EXPECT_EQ(mpi::experimental::type(v2::inplace), MPI_DATATYPE_NULL);
}

// ── null_buf_t ────────────────────────────────────────────────────────────────

static_assert(mpi::experimental::send_buffer<v2::null_buf_t>);
static_assert(mpi::experimental::recv_buffer<v2::null_buf_t>);

TEST(SentinelsTest, NullBufData) {
    EXPECT_EQ(mpi::experimental::ptr(v2::null_buf), nullptr);
    EXPECT_EQ(mpi::experimental::count(v2::null_buf), 0);
    EXPECT_EQ(mpi::experimental::type(v2::null_buf), MPI_DATATYPE_NULL);
}

// ── bottom_t ──────────────────────────────────────────────────────────────────

// bottom_t alone is not a data_buffer (no size or type)
static_assert(!mpi::experimental::data_buffer<v2::bottom_t>);
static_assert(mpi::experimental::has_mpi_ptr<v2::bottom_t>);
static_assert(!mpi::experimental::has_mpi_count<v2::bottom_t>);
static_assert(!mpi::experimental::has_mpi_type<v2::bottom_t>);

TEST(SentinelsTest, BottomData) {
    EXPECT_EQ(mpi::experimental::ptr(v2::bottom), MPI_BOTTOM);
}

// Composed with with_type + with_size it becomes a full send_buffer
TEST(SentinelsTest, BottomComposed) {
    auto buf = v2::bottom | v2::views::with_type(MPI_BYTE) | v2::views::with_size(4);
    static_assert(mpi::experimental::send_buffer<decltype(buf)>);
    EXPECT_EQ(mpi::experimental::ptr(buf), MPI_BOTTOM);
    EXPECT_EQ(mpi::experimental::type(buf), MPI_BYTE);
    EXPECT_EQ(mpi::experimental::count(buf), 4);
}
