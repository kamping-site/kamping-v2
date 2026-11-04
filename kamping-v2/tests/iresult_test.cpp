// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/handle.hpp"
#include "kamping/v2/iresult.hpp"
#include "mpi/handle.hpp"

namespace {

/// Order in which requests completed and owned buffers were destroyed.
std::vector<std::string> events;

/// An owned buffer that logs its destruction (moved-from instances stay silent).
struct tracked_buffer : std::vector<int> {
    std::string name;

    explicit tracked_buffer(std::string n) : std::vector<int>(1), name(std::move(n)) {}
    tracked_buffer(tracked_buffer&& other) noexcept
        : std::vector<int>(std::move(other)),
          name(std::exchange(other.name, {})) {}
    tracked_buffer& operator=(tracked_buffer&& other) noexcept {
        std::vector<int>::operator=(std::move(other));
        name = std::exchange(other.name, {});
        return *this;
    }
    ~tracked_buffer() {
        if (!name.empty()) {
            events.push_back(name + " destroyed");
        }
    }
};

// Generalized-request callbacks: free_fn runs when MPI_Wait/MPI_Test completes and frees the request.
int query_fn(void*, MPI_Status* status) {
    MPI_Status_set_elements(status, MPI_BYTE, 0);
    MPI_Status_set_cancelled(status, 0);
    status->MPI_SOURCE = MPI_UNDEFINED;
    status->MPI_TAG    = MPI_UNDEFINED;
    return MPI_SUCCESS;
}
int free_fn(void* extra) {
    events.push_back(*static_cast<std::string*>(extra) + " completed");
    return MPI_SUCCESS;
}
int cancel_fn(void*, int) {
    return MPI_SUCCESS;
}

/// Fills `*request` with an already-complete generalized request that logs "<name> completed" when freed.
void start_logged_request(MPI_Request* request, std::string& name) {
    MPI_Grequest_start(query_fn, free_fn, cancel_fn, &name, request);
    MPI_Grequest_complete(*request);
}

} // namespace

// ── Lifetime ─────────────────────────────────────────────────────────────────
// Destroying or move-assigning over an iresult with an outstanding request terminates; those scenarios run in
// separate processes (terminate_on_active_request_test.cpp). Here: completed iresults release their buffers normally.

TEST(IresultTest, MoveAssignmentOverCompletedIresultReleasesOldBuffer) {
    events.clear();
    std::string                          old_name = "old request", new_name = "new request";
    kamping::v2::iresult<tracked_buffer> target(tracked_buffer{"old buffer"});
    start_logged_request(kamping::v2::handle_ptr(target), old_name);
    while (!target.test()) {
    }
    events.clear();

    kamping::v2::iresult<tracked_buffer> source(tracked_buffer{"new buffer"});
    start_logged_request(kamping::v2::handle_ptr(source), new_name);
    target = std::move(source);
    EXPECT_TRUE(events.empty()); // the old buffer was already moved out by test()

    auto buf = target.wait();
    EXPECT_EQ(buf.name, "new buffer");
    EXPECT_EQ(events, (std::vector<std::string>{"new request completed"}));
}

TEST(IresultTest, SendRecvWaitThenReleaseBuffers) {
    events.clear();
    std::string request_name = "request";
    {
        kamping::v2::iresult<tracked_buffer, tracked_buffer> res(tracked_buffer{"send"}, tracked_buffer{"recv"});
        start_logged_request(kamping::v2::handle_ptr(res), request_name);
        auto result = res.wait();
        EXPECT_EQ(result.recv.name, "recv");
        EXPECT_EQ(events, (std::vector<std::string>{"request completed"}));
    }
    EXPECT_EQ(events.size(), 3u);
}

// ── Completion ───────────────────────────────────────────────────────────────

static_assert(kamping::v2::convertible_to_mpi_handle_ptr<kamping::v2::iresult<std::vector<int>>, MPI_Request>);
static_assert(mpi::experimental::
                  convertible_to_mpi_handle_ptr<kamping::v2::iresult<std::vector<int>, std::vector<int>>, MPI_Request>);

TEST(IresultTest, WaitReturnsOwnedBufferAndDestructorDoesNotWaitAgain) {
    events.clear();
    std::string request_name = "request";
    {
        kamping::v2::iresult<tracked_buffer> res(tracked_buffer{"buffer"});
        start_logged_request(kamping::v2::handle_ptr(res), request_name);
        tracked_buffer buf = res.wait();
        EXPECT_EQ(buf.name, "buffer");
        EXPECT_EQ(events, (std::vector<std::string>{"request completed"}));
    }
    EXPECT_EQ(events, (std::vector<std::string>{"request completed", "buffer destroyed"}));
}

TEST(IresultTest, TestReturnsOwnedBufferOnCompletion) {
    events.clear();
    std::string                          request_name = "request";
    kamping::v2::iresult<tracked_buffer> res(tracked_buffer{"buffer"});
    start_logged_request(kamping::v2::handle_ptr(res), request_name);
    std::optional<tracked_buffer> buf;
    while (!(buf = res.test())) {
    }
    EXPECT_EQ(buf->name, "buffer");
}

TEST(IresultTest, BorrowedBufferTestReturnsBool) {
    std::vector<int>                        data(1);
    std::string                             request_name = "request";
    kamping::v2::iresult<std::vector<int>&> res(data);
    start_logged_request(kamping::v2::handle_ptr(res), request_name);
    while (!res.test()) {
    }
    EXPECT_EQ(&res.wait(), &data);
}
