// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <ranges>
#include <vector>

#include <gtest/gtest.h>

#include "kamping/v2/views/auto_counts_view.hpp"
#include "kamping/v2/views/auto_displs_view.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/resize_v_view.hpp"
#include "kamping/v2/views/resize_view.hpp"
#include "kamping/v2/views/with_type_view.hpp"

namespace views = kamping::v2::views;

// ── No-op on plain / non-deferred buffers ─────────────────────────────────────

TEST(MaterializeTest, NoOpOnPlainBuffer) {
    std::vector<int> data{1, 2, 3};
    // A plain container is not materializable: the call must compile and change nothing.
    static_assert(!kamping::v2::has_materialize<std::vector<int>>);
    kamping::v2::materialize(data);
    EXPECT_EQ(data.size(), 3u);
}

// ── resize_view (single-count deferred recv) ──────────────────────────────────

TEST(MaterializeTest, ResizeViewIsMaterializable) {
    std::vector<int> data;
    auto             buf = data | views::resize;
    static_assert(kamping::v2::has_materialize<decltype(buf)>);
    static_assert(kamping::v2::deferred_recv_buf<decltype(buf)>);
}

TEST(MaterializeTest, ResizeViewRealizesBeforePtrAccess) {
    std::vector<int> data;
    auto             buf = data | views::resize;

    buf.set_recv_count(5);
    EXPECT_EQ(data.size(), 0u); // deferred — not resized yet

    kamping::v2::materialize(buf);
    EXPECT_EQ(data.size(), 5u); // realized without ever calling mpi_ptr()
}

TEST(MaterializeTest, ResizeViewIsIdempotent) {
    std::vector<int> data;
    auto             buf = data | views::resize;

    buf.set_recv_count(4);
    kamping::v2::materialize(buf);
    kamping::v2::materialize(buf); // second call must not double-resize or crash
    EXPECT_EQ(data.size(), 4u);
    EXPECT_EQ(mpi::experimental::ptr(buf), data.data());
}

// ── resize_v_view (variadic deferred recv) ────────────────────────────────────

TEST(MaterializeTest, ResizeVViewIsMaterializable) {
    std::vector<int> data;
    auto             rbuf = data | views::auto_counts() | views::auto_displs() | views::resize_v;
    static_assert(kamping::v2::has_materialize<decltype(rbuf)>);
    static_assert(kamping::v2::deferred_recv_buf_v<decltype(rbuf)>);
}

TEST(MaterializeTest, ResizeVViewRealizesSizeAndDisplsBeforePtrAccess) {
    std::vector<int> data;
    auto             rbuf = data | views::auto_counts() | views::auto_displs() | views::resize_v;

    rbuf.set_comm_size(3);
    {
        auto* p = std::ranges::data(mpi::experimental::counts(rbuf));
        p[0]    = 2;
        p[1]    = 3;
        p[2]    = 1;
    }
    rbuf.commit_counts();

    kamping::v2::materialize(rbuf);

    // Backing storage sized to the total, and the displacements are readable —
    // exactly what a custom collective needs before laying out the exchange.
    EXPECT_EQ(data.size(), 6u);
    auto displs = mpi::experimental::displs(rbuf);
    EXPECT_EQ((std::vector<int>(std::ranges::begin(displs), std::ranges::end(displs))), (std::vector<int>{0, 2, 5}));
}

// ── Propagation through wrapping view layers ──────────────────────────────────

TEST(MaterializeTest, WrappedVariadicDeferredBufferStaysMaterializable) {
    std::vector<int> data;
    // A wrapper (with_type) around the variadic deferred buffer must still forward the
    // deferred protocol *and* materialize — otherwise mpi_ptr() would resize but an
    // explicit materialize() would silently no-op.
    auto wrapped = data | views::auto_counts() | views::auto_displs() | views::resize_v | views::with_type(MPI_INT);
    static_assert(kamping::v2::deferred_recv_buf_v<decltype(wrapped)>);
    static_assert(kamping::v2::has_materialize<decltype(wrapped)>);

    wrapped.set_comm_size(2);
    {
        auto* p = std::ranges::data(mpi::experimental::counts(wrapped));
        p[0]    = 3;
        p[1]    = 2;
    }
    wrapped.commit_counts();

    kamping::v2::materialize(wrapped);
    EXPECT_EQ(data.size(), 5u); // realized through the wrapper, not just via mpi_ptr()
}

TEST(MaterializeTest, WrappedSingleCountDeferredBufferStaysDeferredAndMaterializable) {
    std::vector<int> data;
    // set_recv_count must be forwarded through the wrapper too, or infer() could not
    // size the buffer.
    auto wrapped = data | views::resize | views::with_type(MPI_INT);
    static_assert(kamping::v2::deferred_recv_buf<decltype(wrapped)>);
    static_assert(kamping::v2::has_materialize<decltype(wrapped)>);

    wrapped.set_recv_count(7);
    EXPECT_EQ(data.size(), 0u);
    kamping::v2::materialize(wrapped);
    EXPECT_EQ(data.size(), 7u);
}

TEST(MaterializeTest, ResizeVViewIsIdempotent) {
    std::vector<int> data;
    auto             rbuf = data | views::auto_counts() | views::auto_displs() | views::resize_v;

    rbuf.set_comm_size(2);
    {
        auto* p = std::ranges::data(mpi::experimental::counts(rbuf));
        p[0]    = 4;
        p[1]    = 1;
    }
    rbuf.commit_counts();

    kamping::v2::materialize(rbuf);
    kamping::v2::materialize(rbuf);
    EXPECT_EQ(data.size(), 5u);
}
