#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views/auto_counts_view.hpp"
#include "kamping/v2/views/auto_displs_view.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/resize_v_view.hpp"

// ── Concept checks ────────────────────────────────────────────────────────────

TEST(AutoCountsViewTest, SatisfiesExpectedConcepts) {
    std::vector<int> data{1, 2, 3};

    auto view = data | kamping::v2::views::auto_counts();
    mpi::experimental::ptr(view);
    static_assert(mpi::experimental::has_mpi_counts<decltype(view)>);
    static_assert(mpi::experimental::has_mpi_counts_mutable<decltype(view)>);
    static_assert(kamping::v2::has_set_comm_size<decltype(view)>);
    static_assert(kamping::v2::has_commit_counts<decltype(view)>);
    static_assert(kamping::v2::deferred_recv_buf_v<decltype(view)>);

    // data/count/type forwarded from base: satisfies recv_buffer
    static_assert(mpi::experimental::recv_buffer<decltype(view)>);
}

TEST(AutoCountsViewTest, FullPipelineSatisfiesRecvBufferV) {
    std::vector<int> data;
    auto             rbuf =
        data | kamping::v2::views::auto_counts() | kamping::v2::views::auto_displs() | kamping::v2::views::resize_v;

    static_assert(kamping::v2::deferred_recv_buf_v<decltype(rbuf)>);
    static_assert(mpi::experimental::recv_buffer_v<decltype(rbuf)>);
    static_assert(kamping::v2::has_monotonic_displs<decltype(rbuf)>);
}

// ── set_comm_size ─────────────────────────────────────────────────────────────

TEST(AutoCountsViewTest, SetCommSizeResizesCountsWhenResizeTrue) {
    std::vector<int> data{10, 20, 30};
    auto             view = data | kamping::v2::views::auto_counts();

    EXPECT_EQ(std::ranges::size(mpi::experimental::counts(view)), 0u);
    view.set_comm_size(3);
    EXPECT_EQ(std::ranges::size(mpi::experimental::counts(view)), 3u);
}

TEST(AutoCountsViewTest, SetCommSizeIsNoOpWhenResizeFalse) {
    std::vector<int> data{10, 20, 30};
    std::vector<int> user_counts(3, 0);
    auto             view = data | kamping::v2::views::auto_counts(user_counts);

    view.set_comm_size(99); // must not resize
    EXPECT_EQ(std::ranges::size(mpi::experimental::counts(view)), 3u);
}

// ── mpi_counts() accessor ─────────────────────────────────────────────────────

TEST(AutoCountsViewTest, CountsAccessorReflectsMpiWrites) {
    std::vector<int> data{1, 2, 3, 4, 5, 6};
    auto             view = data | kamping::v2::views::auto_counts();
    view.set_comm_size(3);

    // Simulate MPI writing counts directly into the buffer
    auto* p = std::ranges::data(mpi::experimental::counts(view));
    p[0]    = 1;
    p[1]    = 2;
    p[2]    = 3;

    auto sv = mpi::experimental::counts(std::as_const(view));
    EXPECT_EQ((std::vector<int>(std::ranges::begin(sv), std::ranges::end(sv))), (std::vector<int>{1, 2, 3}));
}

TEST(AutoCountsViewTest, UserProvidedCountsBuffer) {
    std::vector<int> data{10, 20, 30};
    std::vector<int> user_counts{4, 5, 6};
    auto             view = data | kamping::v2::views::auto_counts(user_counts);

    auto sv = mpi::experimental::counts(view);
    EXPECT_EQ((std::vector<int>(std::ranges::begin(sv), std::ranges::end(sv))), (std::vector<int>{4, 5, 6}));

    // Mutation through mpi_counts() writes through to the original vector
    std::ranges::data(mpi::experimental::counts(view))[0] = 99;
    EXPECT_EQ(user_counts[0], 99);
}

TEST(AutoCountsViewTest, UserProvidedCountsBufferWithResize) {
    std::vector<int> data{1, 2};
    std::vector<int> user_counts; // empty, will be resized
    auto             view = data | kamping::v2::views::auto_counts(kamping::v2::resize, user_counts);

    view.set_comm_size(2);
    EXPECT_EQ(std::ranges::size(mpi::experimental::counts(view)), 2u);
}

// ── commit_counts() ───────────────────────────────────────────────────────────

TEST(AutoCountsViewTest, CommitCountsIsCallable) {
    std::vector<int> data{1, 2};
    auto             view = data | kamping::v2::views::auto_counts();
    view.set_comm_size(2);
    auto* p = std::ranges::data(mpi::experimental::counts(view));
    p[0]    = 1;
    p[1]    = 1;
    view.commit_counts(); // must not crash; currently a no-op
}

// ── base() and view_interface forwarding ─────────────────────────────────────

TEST(AutoCountsViewTest, BaseReturnsUnderlyingDataBuffer) {
    std::vector<int> data{10, 20, 30};
    auto             view = data | kamping::v2::views::auto_counts();

    EXPECT_EQ((std::vector<int>(view.begin(), view.end())), (std::vector<int>{10, 20, 30}));
}

TEST(AutoCountsViewTest, MpiCountForwardsFromBase) {
    std::vector<int> data{1, 2, 3, 4};
    auto             view = data | kamping::v2::views::auto_counts();

    EXPECT_EQ(mpi::experimental::count(view), 4u);
}

TEST(AutoCountsViewTest, MpiDataForwardsFromBase) {
    std::vector<int> data{1, 2, 3};
    auto             view = data | kamping::v2::views::auto_counts();

    EXPECT_EQ(mpi::experimental::ptr(view), data.data());
}

// ── auto_displs integration ───────────────────────────────────────────────────

TEST(AutoCountsViewTest, AutoDisplsComputedFromCounts) {
    std::vector<int> data{1, 2, 3, 4, 5, 6};
    auto             view = data | kamping::v2::views::auto_counts();
    view.set_comm_size(3);
    {
        auto* p = std::ranges::data(mpi::experimental::counts(view));
        p[0]    = 1;
        p[1]    = 2;
        p[2]    = 3;
    }
    view.commit_counts();

    auto chained = std::move(view) | kamping::v2::views::auto_displs();
    auto displs  = mpi::experimental::displs(chained);
    EXPECT_EQ((std::vector<int>(std::ranges::begin(displs), std::ranges::end(displs))), (std::vector<int>{0, 1, 3}));
}

TEST(AutoCountsViewTest, ResizeVResizesDataBufferFromCounts) {
    std::vector<int> data;
    auto             rbuf =
        data | kamping::v2::views::auto_counts() | kamping::v2::views::auto_displs() | kamping::v2::views::resize_v;

    rbuf.set_comm_size(3);
    {
        auto* p = std::ranges::data(mpi::experimental::counts(rbuf));
        p[0]    = 2;
        p[1]    = 3;
        p[2]    = 1;
    }
    rbuf.commit_counts();

    (void)rbuf.mpi_ptr();
    EXPECT_EQ(data.size(), 6u);
}
