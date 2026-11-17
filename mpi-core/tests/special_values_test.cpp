// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <array>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/comm.hpp"
#include "mpi/handle.hpp"
#include "mpi/message.hpp"
#include "mpi/p2p/mprobe.hpp"
#include "mpi/p2p/recv.hpp"
#include "mpi/p2p/send.hpp"
#include "mpi/special_values.hpp"
#include "mpi/status.hpp"
#include "mpi/win.hpp"

using namespace mpi::experimental;

namespace {
/// A user rank type that describes itself with a member function.
struct my_rank {
    int           value;
    constexpr int mpi_rank() const noexcept {
        return value;
    }
};

/// A rank type with both a trait specialization and a member: the trait wins.
struct traited_rank {
    constexpr int mpi_rank() const noexcept {
        return 1;
    }
};

template <typename Dest, typename Tag>
concept sendable = requires(std::array<int, 1> const& buf, Dest dest, Tag tag) {
    mpi::experimental::send(buf, dest, tag, MPI_COMM_SELF);
};

template <typename Source, typename Tag>
concept receivable = requires(std::array<int, 1>& buf, Source source, Tag tag) {
    mpi::experimental::recv(buf, source, tag, MPI_COMM_SELF);
};
} // namespace

template <>
struct mpi::experimental::rank_traits<traited_rank> {
    static constexpr int rank(traited_rank const&) {
        return 2;
    }
};

// ── Values ─────────────────────────────────────────────────────────────────────────────────────────────────────────

static_assert(to_rank(any_source) == MPI_ANY_SOURCE);
static_assert(to_rank(proc_null) == MPI_PROC_NULL);
static_assert(to_tag(any_tag) == MPI_ANY_TAG);
static_assert(to_rank(my_rank{3}) == 3);
static_assert(to_rank(traited_rank{}) == 2);

// Role-specific members: a tag value is no rank and vice versa.
static_assert(rank<any_source_t> && rank<proc_null_t> && !rank<any_tag_t>);
static_assert(tag<any_tag_t> && !tag<any_source_t> && !tag<proc_null_t>);
static_assert(!rank<undefined_t> && !tag<undefined_t>);

// ── Role concepts ──────────────────────────────────────────────────────────────────────────────────────────────────

static_assert(sender<any_source_t> && sender<proc_null_t> && sender<int> && sender<my_rank>);
static_assert(destination<proc_null_t> && destination<int> && destination<my_rank>);
static_assert(!destination<any_source_t> && !destination<any_source_t const&>);
static_assert(message_tag<int> && !message_tag<any_tag_t>);

// Sending to any_source or with any_tag does not compile; raw ints are still accepted.
static_assert(sendable<int, int> && sendable<proc_null_t, int>);
static_assert(!sendable<any_source_t, int> && !sendable<int, any_tag_t>);
static_assert(receivable<any_source_t, any_tag_t> && receivable<int, int> && receivable<proc_null_t, int>);

// ── Point-to-point with the special values ─────────────────────────────────────────────────────────────────────────

TEST(SpecialValuesTest, ReceiveFromAnySourceWithAnyTag) {
    std::array<int, 1> const send_buf{42};
    std::array<int, 1>       recv_buf{0};
    MPI_Request              request = MPI_REQUEST_NULL;
    MPI_Isend(send_buf.data(), 1, MPI_INT, 0, 7, MPI_COMM_SELF, &request);
    status st;
    recv(recv_buf, any_source, any_tag, MPI_COMM_SELF, st);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    EXPECT_EQ(recv_buf[0], 42);
    EXPECT_EQ(st.tag(), 7);
    EXPECT_FALSE(st.from_proc_null());
}

TEST(SpecialValuesTest, ProcNullTransfersNothing) {
    std::array<int, 1> buf{5};
    send(buf, proc_null, 0, MPI_COMM_SELF);
    status st;
    recv(buf, proc_null, any_tag, MPI_COMM_SELF, st);
    EXPECT_EQ(buf[0], 5);
    EXPECT_TRUE(st.from_proc_null());
    EXPECT_EQ(st.source(), MPI_PROC_NULL);
    EXPECT_EQ(st.count<int>(), 0);
}

TEST(SpecialValuesTest, MprobeFromProcNullHasNothingToReceive) {
    message m;
    mprobe(proc_null, any_tag, MPI_COMM_SELF, m);
    // MPI_MESSAGE_NO_PROC: no message to receive, nothing to release.
    EXPECT_EQ(mpi::experimental::handle(m), MPI_MESSAGE_NO_PROC);
}

// ── split / split_type ─────────────────────────────────────────────────────────────────────────────────────────────

TEST(SpecialValuesTest, SplitUndefinedGivesNullComm) {
    // Odd ranks opt out; even ranks form a communicator.
    comm_view world = comm_world();
    comm      sub   = world.rank() % 2 == 0 ? world.split(0, world.rank()) : world.split(undefined);
    if (world.rank() % 2 == 0) {
        EXPECT_NE(mpi::experimental::handle(sub), MPI_COMM_NULL);
        EXPECT_EQ(sub.size(), (world.size() + 1) / 2);
    } else {
        EXPECT_EQ(mpi::experimental::handle(sub), MPI_COMM_NULL);
    }

    MPI_Comm raw = MPI_COMM_NULL;
    if (world.rank() == 0) {
        split(MPI_COMM_WORLD, 0, 0, raw);
        EXPECT_NE(raw, MPI_COMM_NULL);
        MPI_Comm_free(&raw);
    } else {
        split(MPI_COMM_WORLD, undefined, raw);
        EXPECT_EQ(raw, MPI_COMM_NULL);
    }
}

TEST(SpecialValuesTest, SplitTypeShared) {
    comm_view world = comm_world();
    comm      node  = world.split_type(comm_type::shared, world.rank());
    EXPECT_NE(mpi::experimental::handle(node), MPI_COMM_NULL);
    EXPECT_GE(node.size(), 1);
    EXPECT_LE(node.size(), world.size());

    comm free_form = split_type(MPI_COMM_WORLD, comm_type::shared);
    EXPECT_EQ(free_form.size(), node.size());

    comm none = split_type(MPI_COMM_WORLD, undefined);
    EXPECT_EQ(mpi::experimental::handle(none), MPI_COMM_NULL);
}

// ── Window assertions ──────────────────────────────────────────────────────────────────────────────────────────────

static_assert(
    (win_assert::noprecede | win_assert::nostore) == static_cast<win_assert>(MPI_MODE_NOPRECEDE | MPI_MODE_NOSTORE)
);

TEST(SpecialValuesTest, FenceWithTypedAssertions) {
    comm_view world = comm_world();
    auto [w, local] = win_allocate<int>(1, MPI_COMM_WORLD);
    local[0]        = 0;
    w.fence(win_assert::noprecede);
    int const one = 1;
    MPI_Accumulate(&one, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM, mpi::experimental::handle(w));
    fence(w, win_assert::nosucceed | win_assert::nostore);
    if (world.rank() == 0) {
        EXPECT_EQ(local[0], world.size());
    }
    w.fence(0); // plain ints are still accepted
}
