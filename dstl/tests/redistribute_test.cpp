// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dstl/redistribute.hpp"
#include "kamping/v2/comm.hpp"

// ── helpers ───────────────────────────────────────────────────────────────────

namespace {
std::vector<int> iota_vec(int start, int n) {
    std::vector<int> v(static_cast<std::size_t>(n));
    std::iota(v.begin(), v.end(), start);
    return v;
}
} // namespace

// ── Identity ──────────────────────────────────────────────────────────────────
// Same count per rank; every rank keeps its own slice. Exercises self-send.

TEST(Redistribute, Identity) {
    kamping::v2::comm_view comm    = kamping::v2::comm_world();
    int const              rank    = comm.rank();
    int const              local_n = 4;

    auto sbuf = iota_vec(rank * local_n, local_n);
    auto rbuf = std::vector<int>(static_cast<std::size_t>(local_n), -1);

    dstl::redistribute(sbuf, rbuf, comm);

    EXPECT_EQ(rbuf, iota_vec(rank * local_n, local_n));
}

// ── Gather ────────────────────────────────────────────────────────────────────
// All ranks send local_n elements; rank 0 receives everything, others nothing.

TEST(Redistribute, Gather) {
    kamping::v2::comm_view comm    = kamping::v2::comm_world();
    int const              rank    = comm.rank();
    int const              p       = comm.size();
    int const              local_n = 3;

    auto sbuf = iota_vec(rank * local_n, local_n);

    int const recv_n = (rank == 0) ? p * local_n : 0;
    auto      rbuf   = std::vector<int>(static_cast<std::size_t>(recv_n), -1);

    dstl::redistribute(sbuf, rbuf, comm);

    if (rank == 0) {
        EXPECT_EQ(rbuf, iota_vec(0, p * local_n));
    } else {
        EXPECT_TRUE(rbuf.empty());
    }
}

// ── Scatter ───────────────────────────────────────────────────────────────────
// Rank 0 sends p*local_n elements; all ranks receive local_n.

TEST(Redistribute, Scatter) {
    kamping::v2::comm_view comm    = kamping::v2::comm_world();
    int const              rank    = comm.rank();
    int const              p       = comm.size();
    int const              local_n = 3;

    int const send_n = (rank == 0) ? p * local_n : 0;
    auto      sbuf   = iota_vec(0, send_n); // only non-empty on rank 0

    auto rbuf = std::vector<int>(static_cast<std::size_t>(local_n), -1);

    dstl::redistribute(sbuf, rbuf, comm);

    EXPECT_EQ(rbuf, iota_vec(rank * local_n, local_n));
}

// ── EmptyRecvWindowInside ────────────────────────────────────────────────────
// Rank 0 sends everything, only even ranks receive (4 each). With p >= 4, rank 0's send window spans the empty recv
// windows of odd ranks, which must get no (zero-length) message: they post no irecv, so it would stay unmatched.

TEST(Redistribute, EmptyRecvWindowInside) {
    kamping::v2::comm_view comm = kamping::v2::comm_world();
    int const              rank = comm.rank();
    int const              p    = comm.size();
    int const              even = (p + 1) / 2;

    auto sbuf = iota_vec(0, rank == 0 ? 4 * even : 0);
    auto rbuf = std::vector<int>(rank % 2 == 0 ? 4 : 0, -1);

    dstl::redistribute(sbuf, rbuf, comm);

    EXPECT_EQ(rbuf, iota_vec(rank / 2 * 4, rank % 2 == 0 ? 4 : 0));
    // nothing left over for a later exchange with the same tag
    MPI_Barrier(MPI_COMM_WORLD);
    int        pending = 0;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &pending, &status);
    EXPECT_FALSE(pending);
}

// ── EmptySendWindowInside ────────────────────────────────────────────────────
// Only even ranks send (4 each), rank 0 receives everything. With p >= 4, rank 0's recv window spans the empty send
// windows of odd ranks, which post no isend, so a zero-length irecv from them would never complete.

TEST(Redistribute, EmptySendWindowInside) {
    kamping::v2::comm_view comm = kamping::v2::comm_world();
    int const              rank = comm.rank();
    int const              p    = comm.size();
    int const              even = (p + 1) / 2;

    auto sbuf = iota_vec(rank / 2 * 4, rank % 2 == 0 ? 4 : 0);
    auto rbuf = std::vector<int>(rank == 0 ? 4 * even : 0, -1);

    dstl::redistribute(sbuf, rbuf, comm);

    EXPECT_EQ(rbuf, iota_vec(0, rank == 0 ? 4 * even : 0));
}

// ── Uneven ────────────────────────────────────────────────────────────────────
// Rank 0 sends 6, rank 1 sends 2, both receive 4.
// Exercises a cross-rank transfer where the send and recv partitions differ.

TEST(Redistribute, Uneven) {
    kamping::v2::comm_view comm = kamping::v2::comm_world();
    if (comm.size() != 2) {
        GTEST_SKIP() << "2 ranks only";
    }
    int const rank = comm.rank();

    auto sbuf = (rank == 0) ? iota_vec(0, 6) : iota_vec(6, 2);
    auto rbuf = std::vector<int>(4, -1);

    dstl::redistribute(sbuf, rbuf, comm);

    // Global send array = [0..7]; recv windows: rank 0 → [0..3], rank 1 → [4..7].
    auto expected = (rank == 0) ? iota_vec(0, 4) : iota_vec(4, 4);
    EXPECT_EQ(rbuf, expected);
}

// ── SingleRank ────────────────────────────────────────────────────────────────
// p == 1: all data is local, no MPI communication.

TEST(Redistribute, SingleRank) {
    kamping::v2::comm_view comm = kamping::v2::comm_world();
    if (comm.size() != 1) {
        GTEST_SKIP() << "single-rank only";
    }
    std::vector<int> sbuf = {1, 2, 3, 4, 5};
    std::vector<int> rbuf(5, -1);

    dstl::redistribute(sbuf, rbuf, comm);

    EXPECT_EQ(rbuf, sbuf);
}
