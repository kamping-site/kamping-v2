// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/buffer.hpp"
#include "kamping/v2/collectives/bcast.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/datatype.hpp"
#include "kamping/v2/type_pool.hpp"
#include "kamping/v2/views/with_count_view.hpp"
#include "kamping/v2/views/with_type_view.hpp"

namespace v2    = kamping::v2;
namespace views = kamping::v2::views;

namespace {
/// A fresh, uncommitted contiguous datatype of three ints.
MPI_Datatype make_int3() {
    MPI_Datatype dt = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(3, MPI_INT, &dt);
    return dt;
}

/// The datatype type a `with_type` view stores.
template <typename>
struct stored_type;
template <typename Base, typename Type>
struct stored_type<v2::with_type_view<Base, Type>> {
    using type = Type;
};
template <typename View>
using stored_type_t = typename stored_type<std::remove_cvref_t<View>>::type;

struct Unregistered {
    int a, b, c;
};
} // namespace

// ── with_type: borrowed vs. owned datatype ─────────────────────────────────────────────────────────────────────────

TEST(WithTypeDatatypeTest, LvalueHandlesAreBorrowed) {
    std::vector<int> v(3);
    auto const       dt = v2::datatype::from_native(make_int3());

    auto full = views::with_type(v, dt);
    static_assert(std::same_as<stored_type_t<decltype(full)>, MPI_Datatype>);
    EXPECT_EQ(v2::type(full), dt.mpi_handle());

    // Partial application binds the move-only lvalue by reference.
    auto piped = v | views::with_type(dt);
    static_assert(std::same_as<stored_type_t<decltype(piped)>, MPI_Datatype>);
    EXPECT_EQ(v2::type(piped), dt.mpi_handle());

    auto from_view = v | views::with_type(v2::datatype_view(dt));
    static_assert(std::same_as<stored_type_t<decltype(from_view)>, MPI_Datatype>);
    EXPECT_EQ(v2::type(from_view), dt.mpi_handle());

    auto raw = v | views::with_type(MPI_INT);
    static_assert(std::same_as<stored_type_t<decltype(raw)>, MPI_Datatype>);
    EXPECT_EQ(v2::type(raw), MPI_INT);
}

TEST(WithTypeDatatypeTest, TemporaryDatatypeIsOwnedByTheView) {
    std::vector<int> v(3);

    auto piped = v | views::with_type(v2::datatype::from_native(make_int3()));
    static_assert(std::same_as<stored_type_t<decltype(piped)>, v2::datatype>);
    EXPECT_NE(v2::type(piped), MPI_DATATYPE_NULL);
    EXPECT_EQ(v2::datatype_view(v2::type(piped)).size(), 3 * static_cast<int>(sizeof(int)));

    auto full = views::with_type(v, v2::datatype::from_native(make_int3()));
    static_assert(std::same_as<stored_type_t<decltype(full)>, v2::datatype>);

    // A view owning its datatype is never borrowed: the datatype is freed with the view.
    static_assert(!v2::enable_borrowed_buffer<decltype(piped)>);
}

TEST(WithTypeDatatypeTest, BcastWithOwnedDatatype) {
    v2::comm_view const comm(MPI_COMM_WORLD);
    std::vector<int>    v(3, -1);
    if (comm.rank() == 0) {
        v = {1, 2, 3};
    }
    v2::bcast(v | views::with_type(v2::datatype::from_native(make_int3())) | views::with_count(1), 0, comm);
    EXPECT_EQ(v, (std::vector<int>{1, 2, 3}));
}

// ── type_pool with datatype ────────────────────────────────────────────────────────────────────────────────────────

TEST(TypePoolDatatypeTest, RegisterOwnedDatatype) {
    v2::type_pool pool;
    auto          dt  = v2::datatype::from_native(make_int3());
    MPI_Datatype  raw = dt.mpi_handle();

    EXPECT_EQ(pool.register_type<Unregistered>(std::move(dt)), raw);
    EXPECT_EQ(dt.mpi_handle(), MPI_DATATYPE_NULL); // NOLINT(bugprone-use-after-move): the pool took ownership
    ASSERT_TRUE(pool.find<Unregistered>().has_value());
    EXPECT_EQ(*pool.find<Unregistered>(), raw);

    // A second registration keeps the first datatype; the new one stays with the caller.
    auto second = v2::datatype::from_native(make_int3());
    EXPECT_EQ(pool.register_type<Unregistered>(std::move(second)), raw);
    EXPECT_NE(second.mpi_handle(), MPI_DATATYPE_NULL); // NOLINT(bugprone-use-after-move)
}

TEST(TypePoolDatatypeTest, RegisterRawDatatypeCommitsIt) {
    v2::type_pool pool;
    MPI_Datatype  dt = pool.register_type<Unregistered>(make_int3());

    v2::comm_view const comm(MPI_COMM_WORLD);
    std::vector<int>    v(3, -1);
    if (comm.rank() == 0) {
        v = {4, 5, 6};
    }
    v2::bcast(v | views::with_type(dt) | views::with_count(1), 0, comm);
    EXPECT_EQ(v, (std::vector<int>{4, 5, 6}));
}
