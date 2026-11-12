// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/p2p/sendrecv_replace.hpp"
#include "kamping/v2/views.hpp"
#include "mpi/p2p/sendrecv.hpp"
#include "mpi/p2p/sendrecv_replace.hpp"

// ── PMPI intercept ────────────────────────────────────────────────────────────

static int g_sendrecv_replace_calls = 0;

extern "C" {
int MPI_Sendrecv_replace(
    void*        buf,
    int          count,
    MPI_Datatype datatype,
    int          dest,
    int          sendtag,
    int          source,
    int          recvtag,
    MPI_Comm     comm,
    MPI_Status*  status
) {
    ++g_sendrecv_replace_calls;
    return PMPI_Sendrecv_replace(buf, count, datatype, dest, sendtag, source, recvtag, comm, status);
}
}

// ── Deferred buffer with its own infer() ──────────────────────────────────────

namespace custom {
/// A deferred buffer (it has set_recv_count) whose own infer() overload, found by ADL, takes the place of the default
/// one, which rejects deferred buffers. It exchanges the counts and checks that the incoming message has the same
/// length, so it can be received in place.
struct same_size_buf {
    std::vector<int> data;
    int              infer_calls = 0;

    void set_recv_count(std::ptrdiff_t n) {
        data.resize(static_cast<std::size_t>(n));
    }
    std::ptrdiff_t mpi_count() const {
        return static_cast<std::ptrdiff_t>(data.size());
    }
    int* mpi_ptr() {
        return data.data();
    }
    MPI_Datatype mpi_type() const {
        return MPI_INT;
    }
};

void infer(
    kamping::v2::comm_op::sendrecv_replace,
    same_size_buf& buf,
    int            dest,
    int            send_tag,
    int            source,
    int            recv_tag,
    MPI_Comm       comm
) {
    ++buf.infer_calls;
    int send_count = static_cast<int>(buf.data.size());
    int recv_count = -1;
    mpi::experimental::sendrecv(
        kamping::v2::views::ref_single(send_count),
        dest,
        send_tag,
        kamping::v2::views::ref_single(recv_count),
        source,
        recv_tag,
        comm,
        MPI_STATUS_IGNORE
    );
    EXPECT_EQ(recv_count, send_count);
}
} // namespace custom

static_assert(kamping::v2::deferred_recv_buf<custom::same_size_buf>);

// ── MPI functional tests ──────────────────────────────────────────────────────

// Every test shifts data once around a ring: each rank sends to its right neighbor and receives from its left one.
// On one rank, both neighbors are the rank itself.
class V2SendrecvReplaceTest : public ::testing::Test {
protected:
    MPI_Comm comm  = MPI_COMM_WORLD;
    int      rank  = 0;
    int      size  = 0;
    int      left  = 0;
    int      right = 0;

    void SetUp() override {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        left                     = (rank - 1 + size) % size;
        right                    = (rank + 1) % size;
        g_sendrecv_replace_calls = 0;
    }
};

// Full form: the contents of a borrowed buffer are replaced in place, and the buffer is returned by reference.
TEST_F(V2SendrecvReplaceTest, BorrowedBufferWithTags) {
    std::vector<int> buf{rank, 2 * rank, 3 * rank};
    MPI_Status       status;
    // Each rank sends with tag 10 + its rank and receives with tag 10 + the left rank's, so the receive only matches
    // with the tags passed through.
    auto& returned = kamping::v2::sendrecv_replace(buf, right, 10 + rank, left, 10 + left, comm, &status);

    static_assert(std::is_same_v<decltype(returned), std::vector<int>&>);
    EXPECT_EQ(&returned, &buf);
    EXPECT_EQ(buf, (std::vector<int>{left, 2 * left, 3 * left}));
    EXPECT_EQ(status.MPI_SOURCE, left);
    EXPECT_EQ(status.MPI_TAG, 10 + left);
    EXPECT_EQ(g_sendrecv_replace_calls, 1);
}

// Short form with a source: an owned buffer is moved in and handed back by value.
TEST_F(V2SendrecvReplaceTest, OwnedBuffer) {
    auto result = kamping::v2::sendrecv_replace(std::vector<int>(4, rank), right, left, comm);

    static_assert(std::is_same_v<decltype(result), std::vector<int>>);
    EXPECT_EQ(result, std::vector<int>(4, left));
    EXPECT_EQ(g_sendrecv_replace_calls, 1);
}

// Shortest form, receiving from any source: a single value wrapped with views::ref_single. Every rank receives
// exactly one message, so any source is unambiguous.
TEST_F(V2SendrecvReplaceTest, SingleValueFromAnySource) {
    int        item = 100 + rank;
    MPI_Status status;
    kamping::v2::sendrecv_replace(kamping::v2::views::ref_single(item), right, comm, &status);

    EXPECT_EQ(item, 100 + left);
    EXPECT_EQ(status.MPI_SOURCE, left);
    EXPECT_EQ(status.MPI_TAG, kamping::v2::DEFAULT_SEND_TAG);
    EXPECT_EQ(g_sendrecv_replace_calls, 1);
}

// A view that overrides the count: only the first two elements are exchanged, the rest stays untouched.
TEST_F(V2SendrecvReplaceTest, WithCountView) {
    std::vector<int> buf{rank, rank, -1, -1};
    kamping::v2::sendrecv_replace(buf | kamping::v2::views::with_count(2), right, left, comm);

    EXPECT_EQ(buf, (std::vector<int>{left, left, -1, -1}));
    EXPECT_EQ(g_sendrecv_replace_calls, 1);
}

// The core wrapper, which mirrors the MPI signature.
TEST_F(V2SendrecvReplaceTest, CoreWrapper) {
    std::vector<int> buf{rank, rank};
    mpi::experimental::sendrecv_replace(buf, right, 0, left, 0, comm);

    EXPECT_EQ(buf, std::vector<int>(2, left));
    EXPECT_EQ(g_sendrecv_replace_calls, 1);
}

// A deferred buffer is accepted if an infer() overload for it exists; it runs before MPI_Sendrecv_replace.
TEST_F(V2SendrecvReplaceTest, DeferredBufferWithOwnInfer) {
    custom::same_size_buf buf{{rank, rank, rank}};
    kamping::v2::sendrecv_replace(buf, right, left, comm);

    EXPECT_EQ(buf.infer_calls, 1);
    EXPECT_EQ(buf.data, std::vector<int>(3, left));
    EXPECT_EQ(g_sendrecv_replace_calls, 1);
}
