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

static_assert(convertible_to_mpi_handle<uncommitted_datatype, MPI_Datatype>); // for MPI type constructors
static_assert(!std::is_copy_constructible_v<uncommitted_datatype>);
static_assert(std::is_nothrow_move_constructible_v<uncommitted_datatype>);
// An uncommitted datatype can neither be viewed nor used as the datatype of a buffer.
static_assert(!std::is_convertible_v<uncommitted_datatype const&, datatype_view>);
static_assert(buffer_datatype<datatype>);
static_assert(buffer_datatype<datatype const&>);
static_assert(buffer_datatype<datatype_view>);
static_assert(buffer_datatype<MPI_Datatype>);
static_assert(!buffer_datatype<uncommitted_datatype>);
static_assert(!buffer_datatype<uncommitted_datatype const&>);
// commit() is only callable on an rvalue, so no uncommitted handle of the committed datatype is left behind.
template <typename T>
concept committable = requires(T&& u) { std::forward<T>(u).commit(); };
static_assert(committable<uncommitted_datatype>);
static_assert(!committable<uncommitted_datatype&>);

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

TEST(DatatypeTest, CommitMakesItUsable) {
    auto const dt = datatype::commit(make_int3());
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

TEST(DatatypeTest, CommitAcceptsCommittedType) {
    MPI_Datatype raw = make_int3();
    MPI_Type_commit(&raw);
    auto const dt = datatype::commit(raw);
    EXPECT_EQ(dt.mpi_handle(), raw);
}

TEST(DatatypeTest, MoveTransfersOwnership) {
    auto         a   = datatype::commit(make_int3());
    MPI_Datatype raw = a.mpi_handle();

    datatype b = std::move(a);
    EXPECT_EQ(a.mpi_handle(), MPI_DATATYPE_NULL); // NOLINT(bugprone-use-after-move): moved-from state is specified
    EXPECT_EQ(b.mpi_handle(), raw);

    datatype c = datatype::commit(make_int3());
    c          = std::move(b);                    // frees c's previous datatype
    EXPECT_EQ(b.mpi_handle(), MPI_DATATYPE_NULL); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(c.mpi_handle(), raw);
}

TEST(DatatypeTest, ReleaseGivesUpOwnership) {
    auto         dt  = datatype::commit(make_int3());
    MPI_Datatype raw = dt.release();
    EXPECT_EQ(dt.mpi_handle(), MPI_DATATYPE_NULL);
    EXPECT_NE(raw, MPI_DATATYPE_NULL);
    MPI_Type_free(&raw);
}

TEST(DatatypeTest, ExplicitFree) {
    auto dt = datatype::commit(make_int3());
    dt.free();
    EXPECT_EQ(dt.mpi_handle(), MPI_DATATYPE_NULL);
    dt.free(); // no-op on a null datatype
}

TEST(DatatypeTest, ConvertsToView) {
    auto const          dt = datatype::commit(make_int3());
    datatype_view const v  = dt;
    EXPECT_EQ(v.mpi_handle(), dt.mpi_handle());
    EXPECT_EQ(mpi::experimental::handle(dt), dt.mpi_handle());
}

// ── uncommitted_datatype ───────────────────────────────────────────────────────────────────────────────────────────

TEST(UncommittedDatatypeTest, DefaultAndNullAreNull) {
    uncommitted_datatype const a;
    EXPECT_EQ(a.mpi_handle(), MPI_DATATYPE_NULL);
    auto const b = uncommitted_datatype::from_native(MPI_DATATYPE_NULL);
    EXPECT_EQ(b.mpi_handle(), MPI_DATATYPE_NULL);
}

TEST(UncommittedDatatypeTest, OwnsBuildingBlock) {
    // Every other int of a 6-int array: the vector is only a building block for the resized type and is never
    // committed, but it must be freed.
    MPI_Datatype raw_vector = MPI_DATATYPE_NULL;
    MPI_Type_vector(3, 1, 2, MPI_INT, &raw_vector);
    auto const vector = uncommitted_datatype::from_native(raw_vector);
    EXPECT_EQ(vector.size(), 3 * static_cast<int>(sizeof(int)));

    MPI_Datatype raw_resized = MPI_DATATYPE_NULL;
    MPI_Type_create_resized(vector.mpi_handle(), 0, 6 * static_cast<MPI_Aint>(sizeof(int)), &raw_resized);
    auto const resized = datatype::commit(raw_resized);
    EXPECT_EQ(resized.extent(), 6 * static_cast<MPI_Aint>(sizeof(int)));

    int send[6] = {0, 1, 2, 3, 4, 5};
    int recv[3] = {-1, -1, -1};
    ASSERT_EQ(
        MPI_Sendrecv(send, 1, resized.mpi_handle(), 0, 0, recv, 3, MPI_INT, 0, 0, MPI_COMM_SELF, MPI_STATUS_IGNORE),
        MPI_SUCCESS
    );
    EXPECT_EQ(recv[0], 0);
    EXPECT_EQ(recv[1], 2);
    EXPECT_EQ(recv[2], 4);
}

TEST(UncommittedDatatypeTest, CommitTransfersToDatatype) {
    auto         u   = uncommitted_datatype::from_native(make_int3());
    MPI_Datatype raw = u.mpi_handle();

    datatype const dt = std::move(u).commit();
    EXPECT_EQ(u.mpi_handle(), MPI_DATATYPE_NULL); // NOLINT(bugprone-use-after-move): moved-from state is specified
    EXPECT_EQ(dt.mpi_handle(), raw);

    int send[3] = {1, 2, 3};
    int recv[3] = {-1, -1, -1};
    ASSERT_EQ(
        MPI_Sendrecv(send, 1, dt.mpi_handle(), 0, 0, recv, 1, dt.mpi_handle(), 0, 0, MPI_COMM_SELF, MPI_STATUS_IGNORE),
        MPI_SUCCESS
    );
    EXPECT_EQ(recv[1], 2);
}

TEST(UncommittedDatatypeTest, MoveReleaseFree) {
    auto         a   = uncommitted_datatype::from_native(make_int3());
    MPI_Datatype raw = a.mpi_handle();

    uncommitted_datatype b = std::move(a);
    EXPECT_EQ(a.mpi_handle(), MPI_DATATYPE_NULL); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(b.mpi_handle(), raw);

    auto c = uncommitted_datatype::from_native(make_int3());
    c      = std::move(b); // frees c's previous datatype
    EXPECT_EQ(c.mpi_handle(), raw);

    MPI_Datatype released = c.release();
    EXPECT_EQ(c.mpi_handle(), MPI_DATATYPE_NULL);
    MPI_Type_free(&released);

    auto d = uncommitted_datatype::from_native(make_int3());
    d.free();
    EXPECT_EQ(d.mpi_handle(), MPI_DATATYPE_NULL);
}

// ── type_commit ────────────────────────────────────────────────────────────────────────────────────────

namespace {
/// A user wrapper that exposes a writable handle.
struct my_type {
    MPI_Datatype raw = MPI_DATATYPE_NULL;
    MPI_Datatype mpi_handle() const noexcept {
        return raw;
    }
    MPI_Datatype* mpi_handle_ptr() noexcept {
        return &raw;
    }
};

template <typename T>
concept type_committable = requires(T& t) { mpi::experimental::type_commit(t); };
} // namespace

// The owning types commit themselves: a datatype exposes no writable handle, and an uncommitted_datatype, whose
// writable handle receives the result of a type constructor, is rejected explicitly. Views hold only a copy.
static_assert(type_committable<MPI_Datatype>);
static_assert(type_committable<my_type>);
static_assert(!type_committable<datatype>);
static_assert(!type_committable<uncommitted_datatype>);
static_assert(!type_committable<datatype_view>);

TEST(TypeCommitTest, RawHandle) {
    MPI_Datatype dt = make_int3();
    type_commit(dt);

    int send[3] = {1, 2, 3};
    int recv[3] = {-1, -1, -1};
    ASSERT_EQ(MPI_Sendrecv(send, 1, dt, 0, 0, recv, 1, dt, 0, 0, MPI_COMM_SELF, MPI_STATUS_IGNORE), MPI_SUCCESS);
    EXPECT_EQ(recv[2], 3);

    MPI_Type_free(&dt);
}

TEST(TypeCommitTest, WrapperWithHandlePtr) {
    my_type t{make_int3()};
    type_commit(t);
    EXPECT_EQ(datatype_view(t.raw).size(), 3 * static_cast<int>(sizeof(int)));
    MPI_Type_free(&t.raw);
}
