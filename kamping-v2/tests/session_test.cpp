#include <mpi.h>

#if MPI_VERSION >= 4

    #include <string>

    #include <gtest/gtest.h>

    #include "kamping/v2/session.hpp"
    #include "mpi/comm.hpp"
    #include "mpi/group.hpp"
    #include "mpi/session.hpp"

using namespace mpi::experimental;
using namespace kamping::v2;

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
    ASSERT_TRUE(g.has_value());
    EXPECT_GT(g->size(), 0);
}

TEST(SessionTest, GroupFromPsetWorldContainsSelf) {
    session s;
    auto    g = s.group_from_pset(psets::world);
    ASSERT_TRUE(g.has_value());
    EXPECT_TRUE(g->contains_self());
}

TEST(SessionTest, GroupFromPsetSelfHasSizeOne) {
    session s;
    auto    g = s.group_from_pset(psets::self);
    ASSERT_TRUE(g.has_value());
    EXPECT_EQ(g->size(), 1);
}

TEST(SessionTest, GroupFromPsetUnknownThrows) {
    session s;
    EXPECT_THROW((void)s.group_from_pset("mpi://DOES_NOT_EXIST"), mpi_error);
}

// ── comm_from_pset ────────────────────────────────────────────────────────────

TEST(SessionTest, CommFromPsetWorldReturnsValidComm) {
    session s;
    auto    c = s.comm_from_pset(psets::world);
    ASSERT_TRUE(c.has_value());
    EXPECT_GT(c->size(), 0);
    EXPECT_GE(c->rank(), 0);
}

TEST(SessionTest, CommFromPsetSelfHasSizeOne) {
    session s;
    auto    c = s.comm_from_pset(psets::self);
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->size(), 1);
    EXPECT_EQ(c->rank(), 0);
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
