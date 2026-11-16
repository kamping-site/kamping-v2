// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <concepts>
#include <functional>
#include <span>
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
    auto const       dt = v2::datatype::commit(make_int3());

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

TEST(WithTypeDatatypeTest, UncommittedDatatypeIsRejected) {
    using fn = kamping::v2::detail::with_type_fn;
    static_assert(std::invocable<fn, std::vector<int>&, v2::datatype const&>);
    static_assert(std::invocable<fn, std::vector<int>&, v2::datatype>);
    static_assert(std::invocable<fn, std::vector<int>&, MPI_Datatype>);
    static_assert(!std::invocable<fn, std::vector<int>&, v2::uncommitted_datatype const&>);
    static_assert(!std::invocable<fn, std::vector<int>&, v2::uncommitted_datatype>);
    static_assert(!std::invocable<fn, std::vector<int>&, std::reference_wrapper<v2::uncommitted_datatype const>>);
}

TEST(WithTypeDatatypeTest, TemporaryDatatypeIsOwnedByTheView) {
    std::vector<int> v(3);

    auto piped = v | views::with_type(v2::datatype::commit(make_int3()));
    static_assert(std::same_as<stored_type_t<decltype(piped)>, v2::datatype>);
    EXPECT_NE(v2::type(piped), MPI_DATATYPE_NULL);
    EXPECT_EQ(v2::datatype_view(v2::type(piped)).size(), 3 * static_cast<int>(sizeof(int)));

    auto full = views::with_type(v, v2::datatype::commit(make_int3()));
    static_assert(std::same_as<stored_type_t<decltype(full)>, v2::datatype>);

    // A view owning its datatype is never borrowed: the datatype is freed with the view.
    static_assert(!v2::enable_borrowed_buffer<decltype(piped)>);
}

TEST(WithTypeDatatypeTest, BcastWithOwnedDatatype) {
    v2::comm_view const comm = v2::comm_world();
    std::vector<int>    v(3, -1);
    if (comm.rank() == 0) {
        v = {1, 2, 3};
    }
    v2::bcast(v | views::with_type(v2::datatype::commit(make_int3())) | views::with_count(1), 0, comm);
    EXPECT_EQ(v, (std::vector<int>{1, 2, 3}));
}

TEST(WithTypeDatatypeTest, BcastColumnBuiltWithTypeConstructors) {
    // Column 1 of a 3 x 3 row-major matrix: the vector is only a building block, freed after the call.
    v2::comm_view const comm   = v2::comm_world();
    auto const          column = v2::type_create_resized(v2::type_vector(3, 1, 3, MPI_INT), 0, sizeof(int)).commit();

    std::vector<int> m(9, -1);
    if (comm.rank() == 0) {
        m = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    }
    v2::bcast(std::span(m).subspan(1) | views::with_type(column) | views::with_count(1), 0, comm);
    if (comm.rank() != 0) {
        EXPECT_EQ(m, (std::vector<int>{-1, 1, -1, -1, 4, -1, -1, 7, -1}));
    }
}

// ── type_pool with datatype ────────────────────────────────────────────────────────────────────────────────────────

TEST(TypePoolDatatypeTest, RegisterOwnedDatatype) {
    v2::type_pool pool;
    auto          dt  = v2::datatype::commit(make_int3());
    MPI_Datatype  raw = dt.mpi_handle();

    EXPECT_EQ(pool.register_type<Unregistered>(std::move(dt)), raw);
    EXPECT_EQ(dt.mpi_handle(), MPI_DATATYPE_NULL); // NOLINT(bugprone-use-after-move): the pool took ownership
    ASSERT_TRUE(pool.find<Unregistered>().has_value());
    EXPECT_EQ(*pool.find<Unregistered>(), raw);

    // A second registration keeps the first datatype; the new one stays with the caller.
    auto second = v2::datatype::commit(make_int3());
    EXPECT_EQ(pool.register_type<Unregistered>(std::move(second)), raw);
    EXPECT_NE(second.mpi_handle(), MPI_DATATYPE_NULL); // NOLINT(bugprone-use-after-move)
}

TEST(TypePoolDatatypeTest, RegisterCommittedBuildingBlockResult) {
    v2::type_pool pool;
    auto          u  = v2::uncommitted_datatype::from_native(make_int3());
    MPI_Datatype  dt = pool.register_type<Unregistered>(std::move(u).commit());

    v2::comm_view const comm = v2::comm_world();
    std::vector<int>    v(3, -1);
    if (comm.rank() == 0) {
        v = {7, 8, 9};
    }
    v2::bcast(v | views::with_type(dt) | views::with_count(1), 0, comm);
    EXPECT_EQ(v, (std::vector<int>{7, 8, 9}));
}
