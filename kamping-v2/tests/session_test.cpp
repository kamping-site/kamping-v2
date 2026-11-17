// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <mpi.h>

#if MPI_VERSION >= 4

    #include <string>

    #include <gtest/gtest.h>

    #include "kamping/v2/environment.hpp"
    #include "kamping/v2/session.hpp"
    #include "mpi/comm.hpp"
    #include "mpi/group.hpp"
    #include "mpi/info.hpp"
    #include "mpi/session.hpp"

using namespace mpi::experimental;
using namespace kamping::v2;

// ── thread support level ──────────────────────────────────────────────────────

TEST(SessionTest, RequestedThreadLevelIsReportedAsProvided) {
    // MPI may provide a lower level than requested, but always at least MPI_THREAD_SINGLE.
    session s(thread_level::serialized);
    auto    provided = s.provided_thread_level();
    if (provided.has_value()) {
        EXPECT_LE(*provided, thread_level::multiple);
        EXPECT_TRUE(s.info().get("thread_level").has_value());
    }
}

TEST(SessionTest, ConstructFromInfo) {
    info hints;
    hints.set("thread_level", thread_level::funneled);
    session s(hints);
    EXPECT_NE(s.mpi_handle(), MPI_SESSION_NULL);
    auto provided = s.provided_thread_level();
    if (provided.has_value()) {
        EXPECT_GE(*provided, thread_level::single);
    }
}

TEST(SessionTest, ProvidedThreadLevelOnView) {
    session            s(thread_level::single);
    session_view const v = s;
    EXPECT_EQ(v.provided_thread_level(), s.provided_thread_level());
}

TEST(EnvironmentTest, ProvidedThreadLevelMatchesQueryThread) {
    int raw = -1;
    MPI_Query_thread(&raw);
    EXPECT_EQ(static_cast<int>(environment::provided_thread_level()), raw);
}

// ── session construction / move ───────────────────────────────────────────────

TEST(SessionTest, DefaultConstructCreatesValidSession) {
    session s;
    EXPECT_NE(s.mpi_handle(), MPI_SESSION_NULL);
}

TEST(SessionTest, MoveConstructTransfersOwnership) {
    session     a;
    MPI_Session raw = a.mpi_handle();
    session     b   = std::move(a);

    EXPECT_EQ(a.mpi_handle(), MPI_SESSION_NULL);
    EXPECT_EQ(b.mpi_handle(), raw);
}

TEST(SessionTest, MoveAssignTransfersOwnership) {
    session     a;
    session     b;
    MPI_Session raw = a.mpi_handle();
    b               = std::move(a);

    EXPECT_EQ(a.mpi_handle(), MPI_SESSION_NULL);
    EXPECT_EQ(b.mpi_handle(), raw);
}

TEST(SessionTest, ExplicitFinalizeIdempotent) {
    session s;
    s.finalize();
    EXPECT_EQ(s.mpi_handle(), MPI_SESSION_NULL);
    // Second finalize is a no-op (not an error).
    s.finalize();
}

TEST(SessionTest, ReleaseRelinquishesOwnership) {
    session     s;
    MPI_Session raw      = s.mpi_handle();
    MPI_Session released = s.release();

    EXPECT_EQ(released, raw);
    EXPECT_EQ(s.mpi_handle(), MPI_SESSION_NULL);
    MPI_Session_finalize(&released);
}

TEST(SessionTest, FromNativeAdoptsHandle) {
    MPI_Session raw = MPI_SESSION_NULL;
    int         err = MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_RETURN, &raw);
    ASSERT_EQ(err, MPI_SUCCESS);
    session s = session::from_native(raw);
    EXPECT_EQ(s.mpi_handle(), raw);
    // freed by s's destructor
}

TEST(SessionTest, ImplicitConversionToView) {
    session      s;
    session_view sv = s;
    EXPECT_EQ(sv.mpi_handle(), s.mpi_handle());
}

// ── group_from_pset ───────────────────────────────────────────────────────────

TEST(SessionTest, GroupFromPsetWorldReturnsNonEmpty) {
    session s;
    auto    g = s.group_from_pset(psets::world);
    EXPECT_GT(g.size(), 0);
}

TEST(SessionTest, GroupFromPsetWorldContainsSelf) {
    session s;
    auto    g = s.group_from_pset(psets::world);
    EXPECT_TRUE(g.contains_self());
}

TEST(SessionTest, GroupFromPsetSelfHasSizeOne) {
    session s;
    auto    g = s.group_from_pset(psets::self);
    EXPECT_EQ(g.size(), 1);
}

// Intel MPI does not return an error for invalid pset names,
// but produces a fatal error
// TEST(SessionTest, GroupFromPsetUnknownThrows) {
//     session s;
//     EXPECT_THROW((void)s.group_from_pset("mpi://DOES_NOT_EXIST"), mpi_error);
// }

// ── comm_from_pset ────────────────────────────────────────────────────────────

TEST(SessionTest, CommFromPsetWorldReturnsValidComm) {
    session s;
    auto    c = s.comm_from_pset(psets::world);
    EXPECT_GT(c.size(), 0);
    EXPECT_GE(c.rank(), 0);
}

TEST(SessionTest, CommFromPsetWithTag) {
    session s;
    comm    c = s.comm_from_pset(psets::world, "kamping-v2-test");
    EXPECT_GT(c.size(), 0);
}

TEST(SessionTest, CommFromPsetSelfHasSizeOne) {
    session s;
    auto    c = s.comm_from_pset(psets::self);
    EXPECT_EQ(c.size(), 1);
    EXPECT_EQ(c.rank(), 0);
}

// ── psets() range ─────────────────────────────────────────────────────────────

TEST(SessionTest, PsetsRangeIsNonEmpty) {
    session s;
    int     count = 0;
    for ([[maybe_unused]] auto const& name: s.psets()) {
        ++count;
    }
    EXPECT_GT(count, 0);
}

TEST(SessionTest, PsetsRangeContainsWorld) {
    session     s;
    bool        found = false;
    std::string world{psets::world};
    for (auto const& name: s.psets()) {
        if (name == world) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(SessionTest, PsetsRangeContainsSelf) {
    session     s;
    bool        found = false;
    std::string self{psets::self};
    for (auto const& name: s.psets()) {
        if (name == self) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

#endif // MPI_VERSION >= 4
