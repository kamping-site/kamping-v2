// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/comm.hpp"
#include "mpi/handle.hpp"
#include "mpi/p2p/sendrecv.hpp"
#include "mpi/p2p/sendrecv_replace.hpp"
#include "mpi/rank.hpp"

using namespace mpi::experimental;

// ── Concept checks ────────────────────────────────────────────────────────────

static_assert(rank<periodic_rank>);
static_assert(rank<nonperiodic_rank>);
// Converted only via rank_traits, never implicitly to int.
static_assert(!std::is_convertible_v<periodic_rank, int>);
static_assert(!std::is_convertible_v<nonperiodic_rank, int>);

// ── periodic_rank arithmetic ──────────────────────────────────────────────────

static_assert(periodic_rank::with_size(3, 4) + 1 == periodic_rank::with_size(0, 4));
static_assert(periodic_rank::with_size(0, 4) - 1 == periodic_rank::with_size(3, 4));
static_assert((periodic_rank::with_size(1, 4) + 9).value() == 2);
static_assert((periodic_rank::with_size(1, 4) - 9).value() == 0);
static_assert((5 + periodic_rank::with_size(1, 4)).value() == 2);
static_assert(periodic_rank::with_size(-1, 4).value() == 3);
static_assert(periodic_rank::with_size(9, 4).value() == 1);
static_assert(periodic_rank::with_size(0, 1) + 7 == periodic_rank::with_size(0, 1));
static_assert(periodic_rank::with_size(2, 4) != periodic_rank::with_size(2, 5));
static_assert(to_rank(periodic_rank::with_size(3, 4) + 2) == 1);
// Extreme offsets do not overflow.
static_assert((periodic_rank::with_size(5, 7) + std::numeric_limits<int>::max()).value() == 6);
static_assert((periodic_rank::with_size(5, 7) - std::numeric_limits<int>::min()).value() == 0);

constexpr periodic_rank compound_periodic() {
    auto r = periodic_rank::with_size(0, 3);
    r += 4;
    r -= 2;
    return r;
}
static_assert(compound_periodic().value() == 2);

// ── nonperiodic_rank arithmetic ───────────────────────────────────────────────

static_assert((nonperiodic_rank::with_size(2, 4) + 1).value() == 3);
static_assert((nonperiodic_rank::with_size(3, 4) + 1).value() == MPI_PROC_NULL);
static_assert((nonperiodic_rank::with_size(0, 4) - 1).value() == MPI_PROC_NULL);
static_assert((nonperiodic_rank::with_size(0, 4) - 1).is_proc_null());
static_assert(!nonperiodic_rank::with_size(0, 4).is_proc_null());
// The position is kept, so stepping back in brings back the rank.
static_assert((nonperiodic_rank::with_size(3, 4) + 1 - 1).value() == 3);
static_assert((nonperiodic_rank::with_size(0, 4) - 5 + 6).value() == 1);
// All positions outside stand for MPI_PROC_NULL and compare equal.
static_assert(nonperiodic_rank::with_size(-1, 4) == nonperiodic_rank::with_size(7, 4));
static_assert(nonperiodic_rank::with_size(1, 4) != nonperiodic_rank::with_size(2, 4));
static_assert(to_rank(nonperiodic_rank::with_size(5, 4)) == MPI_PROC_NULL);
// Extreme offsets do not overflow the stored position.
static_assert((nonperiodic_rank::with_size(3, 4) + std::numeric_limits<int>::max()).is_proc_null());
static_assert((nonperiodic_rank::with_size(0, 4) + std::numeric_limits<int>::min()).is_proc_null());

// ── MPI functional tests ──────────────────────────────────────────────────────

class RankTest : public ::testing::Test {
protected:
    comm_view comm{MPI_COMM_WORLD};
    int       me   = comm.rank();
    int       size = comm.size();
};

TEST_F(RankTest, ConstructedFromCommunicator) {
    periodic_rank const    p(comm);
    nonperiodic_rank const n(MPI_COMM_WORLD);
    EXPECT_EQ(p.value(), me);
    EXPECT_EQ(p.size(), size);
    EXPECT_EQ(n.value(), me);
    EXPECT_EQ(n.size(), size);
}

// A given rank in a communicator: periodic ranks are wrapped, nonperiodic ones keep their position.
TEST_F(RankTest, ConstructedFromRankAndCommunicator) {
    EXPECT_EQ(periodic_rank(size + 1, comm), periodic_rank::with_size(1, size));
    EXPECT_EQ(periodic_rank(-1, MPI_COMM_WORLD).value(), size - 1);
    EXPECT_EQ(nonperiodic_rank(0, comm).value(), 0);
    EXPECT_TRUE(nonperiodic_rank(size, comm).is_proc_null());
    EXPECT_EQ((nonperiodic_rank(size, comm) - 1).value(), size - 1);
}

// Shifting around the ring: every rank receives the value of its left neighbor.
TEST_F(RankTest, PeriodicRingShift) {
    auto const       r = periodic_rank(comm);
    std::vector<int> buf{me, 2 * me};
    sendrecv_replace(buf, r + 1, 0, r - 1, 0, comm);

    int const left = (me - 1 + size) % size;
    EXPECT_EQ(buf, (std::vector<int>{left, 2 * left}));
}

// Shifting along a line: the first rank has no left neighbor and keeps its initial value.
TEST_F(RankTest, NonperiodicLineShift) {
    auto const r    = nonperiodic_rank(comm);
    int        send = me;
    int        recv = -1;
    sendrecv(std::span(&send, 1), r + 1, 0, std::span(&recv, 1), r - 1, 0, comm, MPI_STATUS_IGNORE);

    EXPECT_EQ(recv, me == 0 ? -1 : me - 1);
}
