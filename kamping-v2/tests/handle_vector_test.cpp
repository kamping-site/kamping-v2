// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/completion.hpp"
#include "kamping/v2/request.hpp"
#include "kamping/v2/status.hpp"
#include "mpi/p2p/irecv.hpp"
#include "mpi/p2p/isend.hpp"

namespace me = mpi::experimental;

namespace {

int self() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

/// Posts a self-message with tag `tag` into `requests`: first the irecv into `recv`, then the isend of `send`.
void post_self_message(me::request_vector& requests, int& send, int& recv, int tag) {
    me::irecv(std::span{&recv, 1}, self(), tag, MPI_COMM_WORLD, requests.emplace_back());
    me::isend(std::span{&send, 1}, self(), tag, MPI_COMM_WORLD, requests.emplace_back());
}

bool all_null(me::request_vector& requests) {
    return std::ranges::all_of(requests.mpi_handle_range(), [](MPI_Request r) { return r == MPI_REQUEST_NULL; });
}

} // namespace

// ── request_vector ───────────────────────────────────────────────────────────

static_assert(!std::is_copy_constructible_v<me::request_vector>);
static_assert(std::is_nothrow_move_constructible_v<me::request_vector>);
static_assert(std::is_nothrow_move_assignable_v<me::request_vector>);
static_assert(std::same_as<std::ranges::range_reference_t<me::request_vector>, me::request_view>);
static_assert(std::same_as<kamping::v2::request_vector, me::request_vector>);

TEST(RequestVectorTest, StartsWithNullSlots) {
    me::request_vector requests(3);
    EXPECT_EQ(requests.size(), 3u);
    EXPECT_TRUE(all_null(requests));
    EXPECT_TRUE(me::testall(requests));
}

TEST(RequestVectorTest, EmplaceBackFeedsNonBlockingOperations) {
    me::request_vector requests;
    std::vector<int>   send{10, 11, 12};
    std::vector<int>   recv(3, -1);
    for (std::size_t i = 0; i < send.size(); ++i) {
        post_self_message(requests, send[i], recv[i], static_cast<int>(i));
    }
    EXPECT_EQ(requests.size(), 6u);
    me::waitall(requests);
    EXPECT_TRUE(all_null(requests));
    EXPECT_EQ(recv, send);
}

TEST(RequestVectorTest, ElementViewsCompleteIndividually) {
    me::request_vector requests;
    int                send = 7, recv = -1;
    post_self_message(requests, send, recv, 0);
    requests[0].wait();
    requests.back().wait();
    EXPECT_TRUE(all_null(requests));
    EXPECT_EQ(recv, 7);
}

TEST(RequestVectorTest, IterationYieldsViews) {
    me::request_vector requests;
    int                send = 3, recv = -1;
    post_self_message(requests, send, recv, 0);
    for (me::request_view r: requests) {
        r.wait();
    }
    EXPECT_TRUE(all_null(requests));
    EXPECT_EQ(recv, 3);
}

TEST(RequestVectorTest, WaitanyAndWaitsomeAcceptIt) {
    me::request_vector requests;
    std::vector<int>   send{1, 2}, recv(2, -1);
    post_self_message(requests, send[0], recv[0], 0);
    post_self_message(requests, send[1], recv[1], 1);

    auto index = me::waitany(requests);
    ASSERT_TRUE(index.has_value());
    EXPECT_LT(*index, requests.size());

    std::vector<int> indices(requests.size());
    while (me::waitsome(requests, indices)) {
    }
    EXPECT_TRUE(all_null(requests));
    EXPECT_EQ(recv, send);
}

TEST(RequestVectorTest, EraseNullKeepsActiveRequestsInOrder) {
    me::request_vector requests(2); // two null slots
    int                send = 5, recv = -1;
    post_self_message(requests, send, recv, 0);
    requests.emplace_back(); // another null slot
    ASSERT_EQ(requests.size(), 5u);

    // Before completion, only the never-started slots are removed.
    MPI_Request const irecv = requests.mpi_handle_range()[2];
    MPI_Request const isend = requests.mpi_handle_range()[3];
    EXPECT_EQ(requests.erase_null(), 3u);
    ASSERT_EQ(requests.size(), 2u);
    EXPECT_EQ(requests.mpi_handle_range()[0], irecv);
    EXPECT_EQ(requests.mpi_handle_range()[1], isend);

    me::waitall(requests);
    EXPECT_EQ(requests.erase_null(), 2u);
    EXPECT_TRUE(requests.empty());
    EXPECT_EQ(recv, 5);
}

// Dropping active slots terminates (see terminate_on_active_request_test.cpp); completed or null slots are fine.
TEST(RequestVectorTest, ModifiersDropCompletedRequests) {
    int                send = 1, recv = -1;
    me::request_vector requests;
    post_self_message(requests, send, recv, 0);
    me::waitall(requests);
    requests.clear();
    EXPECT_TRUE(requests.empty());
    EXPECT_EQ(recv, 1);

    recv = -1;
    post_self_message(requests, send, recv, 1);
    requests.emplace_back();
    requests.pop_back(); // a null slot
    requests.resize(4);  // grows with null slots
    me::waitall(requests);
    requests.resize(0);
    EXPECT_TRUE(requests.empty());
    EXPECT_EQ(recv, 1);
}

TEST(RequestVectorTest, MoveTransfersOwnership) {
    int                send = 4, recv = -1;
    me::request_vector requests;
    post_self_message(requests, send, recv, 0);

    me::request_vector moved(std::move(requests));
    EXPECT_TRUE(requests.empty()); // NOLINT(bugprone-use-after-move): moved-from state is specified
    EXPECT_EQ(moved.size(), 2u);

    // Move assignment over a vector whose requests are all completed (or null) is fine.
    me::request_vector target(3);
    target = std::move(moved);
    EXPECT_EQ(target.size(), 2u);

    me::waitall(target);
    EXPECT_EQ(recv, 4);
}

TEST(RequestVectorTest, ReleaseLeavesNullSlotAndKeepsIndices) {
    me::request_vector requests;
    int                send = 2, recv = -1;
    post_self_message(requests, send, recv, 0);
    MPI_Request const expected = requests.mpi_handle_range()[0];

    MPI_Request released = requests.release(0);
    EXPECT_EQ(released, expected);
    ASSERT_EQ(requests.size(), 2u);                              // slot kept ...
    EXPECT_EQ(requests.mpi_handle_range()[0], MPI_REQUEST_NULL); // ... but null

    me::wait(released); // the caller is now responsible
    me::waitall(requests);
    EXPECT_EQ(requests.erase_null(), 2u);
    EXPECT_EQ(recv, 2);
}

TEST(RequestVectorTest, ExtractAndPushBackMoveOwnership) {
    me::request_vector requests;
    int                send = 3, recv = -1;
    post_self_message(requests, send, recv, 0);

    me::request recv_req = requests.extract(0);
    EXPECT_FALSE(recv_req.is_null());
    EXPECT_EQ(requests.mpi_handle_range()[0], MPI_REQUEST_NULL);

    me::request_vector other;
    other.push_back(std::move(recv_req));
    EXPECT_TRUE(recv_req.is_null()); // NOLINT(bugprone-use-after-move): moved-from state is specified
    ASSERT_EQ(other.size(), 1u);

    me::waitall(other);
    me::waitall(requests);
    EXPECT_EQ(recv, 3);
}

// ── request ──────────────────────────────────────────────────────────────────

static_assert(!std::is_copy_constructible_v<me::request>);
static_assert(std::is_nothrow_move_constructible_v<me::request>);
static_assert(std::is_nothrow_move_assignable_v<me::request>);
static_assert(me::convertible_to_mpi_handle_ptr<me::request, MPI_Request>);
static_assert(std::same_as<kamping::v2::request, me::request>);

TEST(RequestTest, DefaultIsNull) {
    me::request r;
    EXPECT_TRUE(r.is_null());
    EXPECT_TRUE(r.test());
    r.wait();
}

TEST(RequestTest, UsableAsRequestOutputParameter) {
    int         send = 5, recv = -1;
    me::request recv_req, send_req;
    me::irecv(std::span{&recv, 1}, self(), 0, MPI_COMM_WORLD, recv_req);
    me::isend(std::span{&send, 1}, self(), 0, MPI_COMM_WORLD, send_req);
    EXPECT_FALSE(recv_req.is_null());

    me::status status;
    me::wait(recv_req, status); // free function
    EXPECT_EQ(status.tag(), 0);
    while (!send_req.test()) { // member
    }
    EXPECT_TRUE(recv_req.is_null());
    EXPECT_TRUE(send_req.is_null());
    EXPECT_EQ(recv, 5);
}

TEST(RequestTest, ReleaseHandsBackRawHandle) {
    int         send = 6, recv = -1;
    MPI_Request raw = MPI_REQUEST_NULL;
    me::irecv(std::span{&recv, 1}, self(), 0, MPI_COMM_WORLD, raw);
    me::request owner{raw};

    MPI_Request back = owner.release();
    EXPECT_EQ(back, raw);
    EXPECT_TRUE(owner.is_null());

    me::request send_req;
    me::isend(std::span{&send, 1}, self(), 0, MPI_COMM_WORLD, send_req);
    me::wait(back);
    send_req.wait();
    EXPECT_EQ(recv, 6);
}

TEST(RequestTest, MoveTransfersOwnership) {
    int         send = 7, recv = -1;
    me::request recv_req;
    me::irecv(std::span{&recv, 1}, self(), 0, MPI_COMM_WORLD, recv_req);
    me::request moved(std::move(recv_req));
    EXPECT_TRUE(recv_req.is_null()); // NOLINT(bugprone-use-after-move): moved-from state is specified
    EXPECT_FALSE(moved.is_null());

    me::request send_req;
    me::isend(std::span{&send, 1}, self(), 0, MPI_COMM_WORLD, send_req);
    send_req.wait();

    // Move assignment over a null request is fine.
    me::request target;
    target = std::move(moved);
    target.wait();
    EXPECT_EQ(recv, 7);
}

TEST(RequestTest, ConvertsToRequestView) {
    me::request      r;
    me::request_view view = r;
    EXPECT_EQ(view.mpi_handle_ptr(), r.mpi_handle_ptr());
    EXPECT_TRUE(view.test());
}

// ── status_vector ────────────────────────────────────────────────────────────

static_assert(std::is_copy_constructible_v<me::status_vector>);
static_assert(std::same_as<std::ranges::range_reference_t<me::status_vector>, me::status_view>);
static_assert(std::same_as<std::ranges::range_reference_t<me::status_vector const>, me::const_status_view>);
static_assert(std::same_as<kamping::v2::status_vector, me::status_vector>);

TEST(StatusVectorTest, WaitallFillsStatuses) {
    me::request_vector requests;
    std::vector<int>   send{1, 2, 3}, recv(3, -1);
    for (std::size_t i = 0; i < send.size(); ++i) {
        post_self_message(requests, send[i], recv[i], static_cast<int>(i));
    }
    me::status_vector statuses(requests.size());
    me::waitall(requests, statuses);
    for (std::size_t i = 0; i < send.size(); ++i) {
        // statuses[2i] belongs to the irecv of message i.
        EXPECT_EQ(statuses[2 * i].source(), self());
        EXPECT_EQ(statuses[2 * i].tag(), static_cast<int>(i));
        EXPECT_EQ(statuses[2 * i].count<int>(), 1);
    }
    EXPECT_EQ(recv, send);
}

TEST(StatusVectorTest, ConstAccessIsReadOnly) {
    me::request_vector requests;
    int                send = 9, recv = -1;
    post_self_message(requests, send, recv, 42);
    me::status_vector statuses(requests.size());
    me::waitall(requests, statuses);

    me::status_vector const& cstatuses = statuses;
    me::const_status_view    first     = cstatuses.front();
    EXPECT_EQ(first.tag(), 42);
    EXPECT_EQ(first.native(), statuses[0].native());
    static_assert(!me::convertible_to_mpi_handle_ptr<me::const_status_view, MPI_Status>);

    std::vector<int> tags;
    for (auto s: cstatuses | std::views::take(1)) {
        tags.push_back(s.tag());
    }
    EXPECT_EQ(tags, std::vector<int>{42});
}

TEST(StatusVectorTest, ViewsWriteThroughToStorage) {
    me::request_vector requests;
    int                send = 6, recv = -1;
    post_self_message(requests, send, recv, 5);
    me::status_vector statuses(2);
    me::wait(requests[1]);              // the isend
    me::wait(requests[0], statuses[1]); // the irecv, status into slot 1 via a status_view
    EXPECT_EQ(statuses.mpi_handle_range()[1].MPI_TAG, 5);
    EXPECT_EQ(recv, 6);
}
