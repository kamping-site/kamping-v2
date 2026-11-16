// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/datatype.hpp"
#include "mpi/handle.hpp"

using namespace mpi::experimental;

// ── Concept and conversion checks ──────────────────────────────────────────────────────────────────────────────────

static_assert(convertible_to_mpi_handle<datatype, MPI_Datatype>);
static_assert(convertible_to_mpi_handle<datatype_view, MPI_Datatype>);
static_assert(!std::is_copy_constructible_v<datatype>);
static_assert(std::is_nothrow_move_constructible_v<datatype>);
static_assert(std::is_convertible_v<MPI_Datatype, datatype_view>);
static_assert(std::is_convertible_v<datatype const&, datatype_view>);
// A view of a temporary would dangle as soon as the temporary frees its datatype.
static_assert(!std::is_convertible_v<datatype&&, datatype_view>);

namespace {
/// A fresh, uncommitted contiguous datatype of three ints.
MPI_Datatype make_int3() {
    MPI_Datatype dt = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(3, MPI_INT, &dt);
    return dt;
}
} // namespace

// ── datatype_view ──────────────────────────────────────────────────────────────────────────────────────────────────

TEST(DatatypeViewTest, AccessorsOfPredefinedType) {
    datatype_view const v = MPI_DOUBLE;
    EXPECT_EQ(v.mpi_handle(), MPI_DOUBLE);
    EXPECT_EQ(v.size(), static_cast<int>(sizeof(double)));
    EXPECT_EQ(v.extent(), static_cast<MPI_Aint>(sizeof(double)));
    EXPECT_EQ(v.lower_bound(), 0);
}

// ── datatype ───────────────────────────────────────────────────────────────────────────────────────────────────────

TEST(DatatypeTest, DefaultIsNull) {
    datatype const dt;
    EXPECT_EQ(dt.mpi_handle(), MPI_DATATYPE_NULL);
}

TEST(DatatypeTest, FromNativeCommitsAndIsUsable) {
    auto const dt = datatype::from_native(make_int3());
    EXPECT_NE(dt.mpi_handle(), MPI_DATATYPE_NULL);
    EXPECT_EQ(dt.size(), 3 * static_cast<int>(sizeof(int)));
    EXPECT_EQ(dt.extent(), 3 * static_cast<MPI_Aint>(sizeof(int)));

    // Only a committed datatype can be used for communication.
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int send[3] = {rank, rank + 1, rank + 2};
    int recv[3] = {-1, -1, -1};
    ASSERT_EQ(
        MPI_Sendrecv(send, 1, dt.mpi_handle(), 0, 0, recv, 1, dt.mpi_handle(), 0, 0, MPI_COMM_SELF, MPI_STATUS_IGNORE),
        MPI_SUCCESS
    );
    EXPECT_EQ(recv[0], rank);
    EXPECT_EQ(recv[2], rank + 2);
}

TEST(DatatypeTest, FromNativeAcceptsCommittedType) {
    MPI_Datatype raw = make_int3();
    MPI_Type_commit(&raw);
    auto const dt = datatype::from_native(raw);
    EXPECT_EQ(dt.mpi_handle(), raw);
}

TEST(DatatypeTest, MoveTransfersOwnership) {
    auto         a   = datatype::from_native(make_int3());
    MPI_Datatype raw = a.mpi_handle();

    datatype b = std::move(a);
    EXPECT_EQ(a.mpi_handle(), MPI_DATATYPE_NULL); // NOLINT(bugprone-use-after-move): moved-from state is specified
    EXPECT_EQ(b.mpi_handle(), raw);

    datatype c = datatype::from_native(make_int3());
    c          = std::move(b);                    // frees c's previous datatype
    EXPECT_EQ(b.mpi_handle(), MPI_DATATYPE_NULL); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(c.mpi_handle(), raw);
}

TEST(DatatypeTest, ReleaseGivesUpOwnership) {
    auto         dt  = datatype::from_native(make_int3());
    MPI_Datatype raw = dt.release();
    EXPECT_EQ(dt.mpi_handle(), MPI_DATATYPE_NULL);
    EXPECT_NE(raw, MPI_DATATYPE_NULL);
    MPI_Type_free(&raw);
}

TEST(DatatypeTest, ExplicitFree) {
    auto dt = datatype::from_native(make_int3());
    dt.free();
    EXPECT_EQ(dt.mpi_handle(), MPI_DATATYPE_NULL);
    dt.free(); // no-op on a null datatype
}

TEST(DatatypeTest, ConvertsToView) {
    auto const          dt = datatype::from_native(make_int3());
    datatype_view const v  = dt;
    EXPECT_EQ(v.mpi_handle(), dt.mpi_handle());
    EXPECT_EQ(mpi::experimental::handle(dt), dt.mpi_handle());
}
