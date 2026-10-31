// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#if MPI_VERSION >= 4
    #include "kamping/v2/p2p/isendrecv.hpp"
    #include "kamping/v2/request.hpp"
    #include "kamping/v2/views.hpp"
#endif


#if MPI_VERSION >= 4

namespace {
/// Pairs ranks 0 and 1; other ranks (and runs on one rank) take no part.
bool paired(int& peer) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    peer = 1 - rank;
    return size >= 2 && rank < 2;
}
} // namespace

// iresult form, deferred receive buffer: the counts are exchanged first, then the data.
TEST(V2IsendrecvTest, DeferredRecvBuffer) {
    int peer;
    if (!paired(peer)) {
        GTEST_SKIP() << "needs ranks 0 and 1";
    }
    std::vector<int> send_data(static_cast<std::size_t>(2 + peer), 1 - peer); // rank 0 sends 3, rank 1 sends 2
    std::vector<int> recv_data;
    auto             pending = kamping::v2::isendrecv(send_data, peer, recv_data | kamping::v2::views::resize, peer, MPI_COMM_WORLD);
    pending.wait();
    EXPECT_EQ(recv_data, std::vector<int>(static_cast<std::size_t>(3 - peer), peer));
}

// iresult form with owned buffers and explicit tags; the buffers are handed back by wait().
TEST(V2IsendrecvTest, OwnedBuffersWithTags) {
    int peer;
    if (!paired(peer)) {
        GTEST_SKIP() << "needs ranks 0 and 1";
    }
    MPI_Status status;
    auto       result = kamping::v2::isendrecv(
                      std::vector<int>{1 - peer},
                      peer,
                      10 + (1 - peer),
                      std::vector<int>{} | kamping::v2::views::resize,
                      peer,
                      10 + peer,
                      MPI_COMM_WORLD
    )
                      .wait(&status);
    EXPECT_EQ(result.send, std::vector<int>{1 - peer});
    EXPECT_EQ(result.recv.underlying(), std::vector<int>{peer});
    EXPECT_EQ(status.MPI_TAG, 10 + peer);
}

// Caller-supplied request, receiving from any source with any tag.
TEST(V2IsendrecvTest, CallerSuppliedRequest) {
    int peer;
    if (!paired(peer)) {
        GTEST_SKIP() << "needs ranks 0 and 1";
    }
    std::vector<int>     send_data{1 - peer, 1 - peer};
    std::vector<int>     recv_data(2);
    kamping::v2::request req;
    kamping::v2::isendrecv(req, send_data, peer, recv_data, MPI_COMM_WORLD);
    req.wait();
    EXPECT_EQ(recv_data, std::vector<int>(2, peer));
}

#else

TEST(V2IsendrecvTest, RequiresMpi4) {
    GTEST_SKIP() << "MPI_Isendrecv requires MPI 4.0";
}

#endif
