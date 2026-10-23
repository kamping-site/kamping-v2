// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/completion.hpp"
#include "mpi/p2p/irecv.hpp"
#include "mpi/p2p/isend.hpp"
#include "mpi/request.hpp"
#include "mpi/sentinels.hpp"
#include "mpi/status.hpp"

namespace me = mpi::experimental;

namespace {

int self() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

/// Posts `n` self-messages: requests[2i] is the irecv and requests[2i + 1] the isend of message i,
/// sent with tag i and payload 100 + i.
struct self_messages {
    std::vector<int>         send;
    std::vector<int>         recv;
    std::vector<MPI_Request> requests;

    explicit self_messages(std::size_t n) : send(n), recv(n, -1), requests(2 * n, MPI_REQUEST_NULL) {
        for (std::size_t i = 0; i < n; ++i) {
            int const tag = static_cast<int>(i);
            send[i]       = 100 + tag;
            me::irecv(std::span{recv}.subspan(i, 1), self(), tag, MPI_COMM_WORLD, requests[2 * i]);
            me::isend(std::span{send}.subspan(i, 1), self(), tag, MPI_COMM_WORLD, requests[2 * i + 1]);
        }
    }

    void expect_received() const {
        for (std::size_t i = 0; i < recv.size(); ++i) {
            EXPECT_EQ(recv[i], 100 + static_cast<int>(i));
        }
    }
};

bool all_null(std::vector<MPI_Request> const& requests) {
    return std::ranges::all_of(requests, [](MPI_Request r) { return r == MPI_REQUEST_NULL; });
}

} // namespace

// ── wait / test ──────────────────────────────────────────────────────────────

TEST(CompletionTest, WaitOnRawRequestAndView) {
    self_messages m(1);
    me::status    status;
    me::wait(m.requests[0], status);
    EXPECT_EQ(status.source(), self());
    EXPECT_EQ(status.tag(), 0);
    me::wait(me::request_view{m.requests[1]});
    EXPECT_TRUE(all_null(m.requests));
    m.expect_received();
}

TEST(CompletionTest, WaitAndTestOnNullRequest) {
    MPI_Request req = MPI_REQUEST_NULL;
    me::wait(req);
    EXPECT_TRUE(me::test(req));
    EXPECT_TRUE(me::test(&req));
}

TEST(CompletionTest, TestPollsUntilDone) {
    self_messages m(1);
    while (!me::test(m.requests[0])) {
    }
    while (!me::test(m.requests[1])) {
    }
    EXPECT_TRUE(all_null(m.requests));
    m.expect_received();
}

TEST(CompletionTest, RequestViewMembersDelegate) {
    self_messages m(1);
    me::request_view{m.requests[0]}.wait();
    while (!me::request_view{m.requests[1]}.test()) {
    }
    EXPECT_TRUE(all_null(m.requests));
    m.expect_received();
}

// ── waitall / testall ────────────────────────────────────────────────────────

TEST(CompletionTest, WaitallCompletesAll) {
    self_messages m(3);
    me::waitall(m.requests);
    EXPECT_TRUE(all_null(m.requests));
    m.expect_received();
}

TEST(CompletionTest, WaitallWithStatuses) {
    self_messages           m(3);
    std::vector<MPI_Status> statuses(m.requests.size());
    me::waitall(m.requests, statuses);
    for (std::size_t i = 0; i < 3; ++i) {
        // statuses[i] belongs to requests[i]; only the recv statuses carry source and tag.
        EXPECT_EQ(statuses[2 * i].MPI_SOURCE, self());
        EXPECT_EQ(statuses[2 * i].MPI_TAG, static_cast<int>(i));
    }
    m.expect_received();
}

TEST(CompletionTest, WaitallOnArrayAndSpan) {
    MPI_Request array[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    me::waitall(array);
    me::waitall(std::span{array});
    std::vector<MPI_Request> empty;
    me::waitall(empty);
}

TEST(CompletionTest, WaitallAndTestallWithExplicitStatusesIgnore) {
    self_messages m(2);
    me::waitall(m.requests, me::statuses_ignore);
    EXPECT_TRUE(all_null(m.requests));
    EXPECT_TRUE(me::testall(m.requests, me::statuses_ignore));
    m.expect_received();
}

TEST(CompletionTest, WaitallWithStatusesFromRawPointerSpan) {
    self_messages m(2);
    MPI_Status    raw[4];
    MPI_Status*   ptr = raw; // memory only known by pointer: wrap it with its length
    me::waitall(m.requests, std::span{ptr, 4});
    EXPECT_EQ(raw[0].MPI_TAG, 0);
    EXPECT_EQ(raw[2].MPI_TAG, 1);
    m.expect_received();
}

TEST(CompletionTest, TestallPollsUntilDone) {
    self_messages           m(3);
    std::vector<MPI_Status> statuses(m.requests.size());
    while (!me::testall(m.requests, statuses)) {
    }
    EXPECT_TRUE(all_null(m.requests));
    EXPECT_EQ(statuses[0].MPI_TAG, 0);
    m.expect_received();
}

// ── waitany / testany ────────────────────────────────────────────────────────

TEST(CompletionTest, WaitanyReturnsEachIndexOnce) {
    self_messages            m(2);
    std::vector<std::size_t> seen;
    while (auto index = me::waitany(m.requests)) {
        ASSERT_LT(*index, m.requests.size());
        EXPECT_EQ(m.requests[*index], MPI_REQUEST_NULL);
        seen.push_back(*index);
    }
    std::ranges::sort(seen);
    EXPECT_EQ(seen, (std::vector<std::size_t>{0, 1, 2, 3}));
    m.expect_received();
}

TEST(CompletionTest, WaitanyWithStatus) {
    self_messages m(1);
    me::status    status;
    // Only the recv request carries a meaningful status; complete the send first.
    me::wait(m.requests[1]);
    auto index = me::waitany(m.requests, status);
    ASSERT_EQ(index, std::optional<std::size_t>{0});
    EXPECT_EQ(status.source(), self());
    EXPECT_EQ(status.tag(), 0);
}

TEST(CompletionTest, WaitanyAndTestanyOnNoActiveRequests) {
    std::vector<MPI_Request> requests(3, MPI_REQUEST_NULL);
    EXPECT_EQ(me::waitany(requests), std::nullopt);
    auto result = me::testany(requests);
    EXPECT_FALSE(result.pending());   // would not block ...
    EXPECT_FALSE(result.has_value()); // ... because nothing is active
    EXPECT_EQ(result.index(), std::nullopt);
    EXPECT_THROW(std::ignore = result.value(), std::bad_optional_access);
}

TEST(CompletionTest, TestanyPollsUntilAllDone) {
    self_messages            m(2);
    std::vector<std::size_t> seen;
    for (;;) {
        auto result = me::testany(m.requests);
        if (result.pending()) {
            continue; // nothing completed yet
        }
        if (!result.has_value()) {
            break; // no active requests left
        }
        EXPECT_EQ(result.value(), *result);
        EXPECT_EQ(result.index(), std::optional<std::size_t>{*result});
        seen.push_back(*result);
    }
    std::ranges::sort(seen);
    EXPECT_EQ(seen, (std::vector<std::size_t>{0, 1, 2, 3}));
    m.expect_received();
}

TEST(CompletionTest, TestanyResultStates) {
    me::testany_result const pending{false, MPI_UNDEFINED};
    EXPECT_TRUE(pending.pending());
    EXPECT_FALSE(pending.has_value());
    EXPECT_THROW(std::ignore = pending.value(), std::bad_optional_access);

    me::testany_result const completed{true, 3};
    EXPECT_FALSE(completed.pending());
    EXPECT_TRUE(completed.has_value());
    EXPECT_EQ(*completed, 3u);
    EXPECT_EQ(completed.index(), std::optional<std::size_t>{3});

    me::testany_result const no_active{true, MPI_UNDEFINED};
    EXPECT_FALSE(no_active.pending());
    EXPECT_FALSE(no_active.has_value());
    EXPECT_EQ(no_active.index(), std::nullopt);

    static_assert(!std::is_constructible_v<bool, me::testany_result>, "no operator bool");
}

// ── waitsome / testsome ──────────────────────────────────────────────────────

TEST(CompletionTest, WaitsomeReportsCompletedIndices) {
    self_messages            m(3);
    std::vector<int>         indices(m.requests.size());
    std::vector<std::size_t> seen;
    while (auto n = me::waitsome(m.requests, indices)) {
        ASSERT_GE(*n, 1u);
        for (std::size_t k = 0; k < *n; ++k) {
            EXPECT_EQ(m.requests[static_cast<std::size_t>(indices[k])], MPI_REQUEST_NULL);
            seen.push_back(static_cast<std::size_t>(indices[k]));
        }
    }
    std::ranges::sort(seen);
    EXPECT_EQ(seen, (std::vector<std::size_t>{0, 1, 2, 3, 4, 5}));
    m.expect_received();
}

TEST(CompletionTest, WaitsomeStatusesAreInCompletionOrder) {
    self_messages m(3);
    // Complete the sends first so that every status written below belongs to a recv.
    for (std::size_t i = 0; i < 3; ++i) {
        me::wait(m.requests[2 * i + 1]);
    }
    std::vector<int>        indices(m.requests.size());
    std::vector<MPI_Status> statuses(m.requests.size());
    while (auto n = me::waitsome(m.requests, indices, statuses)) {
        for (std::size_t k = 0; k < *n; ++k) {
            // statuses[k] belongs to requests[indices[k]], whose message has tag indices[k] / 2.
            EXPECT_EQ(statuses[k].MPI_TAG, indices[k] / 2);
            EXPECT_EQ(statuses[k].MPI_SOURCE, self());
        }
    }
    m.expect_received();
}

TEST(CompletionTest, TestsomePollsUntilAllDone) {
    self_messages            m(3);
    std::array<int, 6>       indices{};
    std::vector<MPI_Status>  statuses(m.requests.size());
    std::vector<std::size_t> seen;
    while (auto n = me::testsome(m.requests, indices, statuses)) {
        for (std::size_t k = 0; k < *n; ++k) { // *n may be 0: nothing completed yet
            seen.push_back(static_cast<std::size_t>(indices[k]));
        }
    }
    std::ranges::sort(seen);
    EXPECT_EQ(seen, (std::vector<std::size_t>{0, 1, 2, 3, 4, 5}));
    m.expect_received();
}

TEST(CompletionTest, WaitsomeWithExplicitStatusesIgnore) {
    self_messages    m(2);
    std::vector<int> indices(m.requests.size());
    std::size_t      completed = 0;
    while (auto n = me::waitsome(m.requests, indices, me::statuses_ignore)) {
        completed += *n;
    }
    EXPECT_EQ(completed, m.requests.size());
    m.expect_received();
}

TEST(CompletionTest, WaitsomeAndTestsomeOnNoActiveRequests) {
    std::vector<MPI_Request> requests(2, MPI_REQUEST_NULL);
    std::vector<int>         indices(2);
    EXPECT_EQ(me::waitsome(requests, indices), std::nullopt);
    EXPECT_EQ(me::testsome(requests, indices), std::nullopt);
}

// ── kamping::v2 re-exports ───────────────────────────────────────────────────

TEST(CompletionTest, V2Reexports) {
    self_messages m(2);
    kamping::v2::waitall(m.requests);
    EXPECT_TRUE(kamping::v2::testall(m.requests));
    m.expect_received();
}
