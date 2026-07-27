// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/types/contiguous_type.hpp"
#include "kamping/types/mpi_type_traits.hpp"
#include "kamping/types/std/unsafe/utility.hpp"
#include "kamping/v2/collectives/alltoallv.hpp"
#include "kamping/v2/type_pool.hpp"
#include "kamping/v2/views.hpp"
#include "kamping/v2/views/flatten_v_view.hpp"
#include "kamping/v2/views/payload.hpp"
#include "kamping/v2/views/with_type_view.hpp"
#include "kamping/v2/views/with_value_type_view.hpp"
#include "mpi/buffer.hpp"

namespace views = kamping::v2::views;

struct MyStruct {
    int    x;
    double y;
};

template <>
struct kamping::types::mpi_type_traits<MyStruct> : public kamping::types::byte_serialized<MyStruct> {};

// Deliberately no kamping::types::mpi_type_traits<TraitlessStruct> specialization anywhere — this is
// the exact case the one-shot register_type<T>(MPI_Datatype) overload exists for, and find<T>() must
// not hard-error on it (mpi_type_traits<T> is the empty primary template, with no members at all).
struct TraitlessStruct {
    int x;
};

// ── payload_element_t: the flat fallback subsumes the old mpi_element_type_t ───────────────

TEST(PayloadElementTest, PlainRangeFallsBackToRangeValueType) {
    static_assert(std::same_as<kamping::v2::payload_element_t<std::vector<int>>, int>);
}

TEST(PayloadElementTest, ValueDestinationPairResolvesToValueSlotOnly) {
    using Buf = std::vector<std::pair<MyStruct, int>>;
    static_assert(std::same_as<kamping::v2::payload_element_t<Buf>, MyStruct>);
}

TEST(PayloadElementTest, NestedResolvesToInnerElement) {
    using Buf = std::vector<std::vector<int>>;
    static_assert(std::same_as<kamping::v2::payload_element_t<Buf>, int>);
}

// ── value_type(): tier-3 fallback (builtin deduction from the payload) ─────────────────────

TEST(ValueTypeTest, BuiltinFallbackForPlainRange) {
    std::vector<int> v{1, 2, 3};
    EXPECT_EQ(kamping::v2::value_type(v), MPI_INT);
}

TEST(ValueTypeTest, BuiltinFallbackForValueDestinationPairs) {
    std::vector<std::pair<int, int>> pairs{{1, 0}, {2, 1}};
    EXPECT_EQ(kamping::v2::value_type(pairs), MPI_INT);
    EXPECT_TRUE((kamping::v2::has_mpi_value_type<std::vector<std::pair<int, int>>>));
}

TEST(ValueTypeTest, NoValueTypeForNonBuiltinUnannotatedPairs) {
    std::vector<std::pair<MyStruct, int>> pairs;
    EXPECT_FALSE((kamping::v2::has_mpi_value_type<std::vector<std::pair<MyStruct, int>>>));
}

// ── with_value_type_view: writes mpi_value_type(), leaves mpi_type() alone ─────────────────

TEST(WithValueTypeViewTest, SetsValueTypeChannelOnly) {
    std::vector<std::pair<MyStruct, int>> pairs{{{1, 2.0}, 0}};
    MPI_Datatype                          dt = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(static_cast<int>(sizeof(MyStruct)), MPI_BYTE, &dt);
    MPI_Type_commit(&dt);

    auto view = pairs | views::with_value_type(dt);
    EXPECT_EQ(kamping::v2::value_type(view), dt);
    EXPECT_FALSE(mpi::experimental::has_mpi_type<decltype(view)>);
    // Still recognizable as a value_destination_pair_buffer: begin/end forwarded transparently.
    EXPECT_EQ(std::ranges::distance(view), 1);
    static_assert(kamping::v2::value_destination_pair_buffer<decltype(view)>);

    MPI_Type_free(&dt);
}

TEST(WithValueTypeViewTest, DisambiguatesFromWithType) {
    // pair<int,int> matches both value_destination_pair (rank in slot 1) and a plain struct — the
    // channel distinguishes "send each pair as a struct" (with_type) from "send each value slot
    // as int" (with_value_type). Both can be attached without conflict.
    std::vector<std::pair<int, int>> pairs{{1, 0}};
    auto                             view = pairs | views::with_value_type(MPI_INT);
    EXPECT_EQ(kamping::v2::value_type(view), MPI_INT);
    // mpi_type() still falls through to the plain-range builtin deduction on the base — of the
    // *pair* value_type — since with_value_type never touches that channel.
}

// ── forwarding through view_interface: annotation survives further wrapping ────────────────

TEST(WithValueTypeViewTest, SurvivesFurtherWrapping) {
    std::vector<std::pair<MyStruct, int>> pairs{{{1, 2.0}, 0}};
    MPI_Datatype                          dt = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(static_cast<int>(sizeof(MyStruct)), MPI_BYTE, &dt);
    MPI_Type_commit(&dt);

    // Wrap the annotated view in a second, unrelated layer (with_type on top, a stand-in for
    // any outer adaptor) and confirm mpi_value_type() still forwards from the inner base.
    auto inner  = pairs | views::with_value_type(dt);
    auto nested = inner | views::with_type(MPI_INT); // unrelated element-channel annotation
    EXPECT_EQ(kamping::v2::value_type(nested), dt);

    MPI_Type_free(&dt);
}

// ── flatten_v_view: honors the with_value_type annotation over the deduced fallback ────────

TEST(FlattenVValueTypeTest, ReportsAnnotatedValueTypeAsFlatBufferType) {
    std::vector<std::pair<MyStruct, int>> pairs{{{1, 2.0}, 0}, {{3, 4.0}, 0}};
    MPI_Datatype                          dt = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(static_cast<int>(sizeof(MyStruct)), MPI_BYTE, &dt);
    MPI_Type_commit(&dt);

    auto flat = pairs | views::with_value_type(dt) | views::flatten_v();
    flat.set_comm_size(1);
    EXPECT_EQ(mpi::experimental::type(flat), dt);

    MPI_Type_free(&dt);
}

TEST(FlattenVValueTypeTest, FallsBackToDeducedFlatTypeWithoutAnnotation) {
    std::vector<std::pair<int, int>> pairs{{1, 0}, {2, 0}};
    auto                             flat = pairs | views::flatten_v();
    flat.set_comm_size(1);
    EXPECT_EQ(mpi::experimental::type(flat), MPI_INT);
}

// ── type_pool: one-shot register_type(dt) + trait-free find<T>() ──────────────────────────

TEST(TypePoolValueTypeTest, OneShotRegisterThenFind) {
    kamping::v2::type_pool pool;
    MPI_Datatype           dt = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(static_cast<int>(sizeof(MyStruct)), MPI_BYTE, &dt);
    // Deliberately NOT pre-committed: register_type(dt) must commit it (ScopedDatatype), like
    // the trait-based overload already does for byte_serialized<T>::data_type().
    MPI_Datatype registered = pool.register_type<MyStruct>(dt);
    ASSERT_NE(registered, MPI_DATATYPE_NULL);
    auto found = pool.find<MyStruct>();
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, registered);
}

TEST(TypePoolValueTypeTest, RedundantOneShotRegistrationKeepsFirstAndFreesSecond) {
    kamping::v2::type_pool pool;
    MPI_Datatype           dt1 = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(static_cast<int>(sizeof(MyStruct)), MPI_BYTE, &dt1);
    MPI_Datatype first = pool.register_type<MyStruct>(dt1);

    // Simulates calling a helper (e.g. one that wraps request_reply) a second time: a *fresh*
    // derived datatype is constructed and handed to register_type again. It must not leak — the
    // pool frees it immediately and keeps returning the original handle.
    MPI_Datatype dt2 = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(static_cast<int>(sizeof(MyStruct)), MPI_BYTE, &dt2);
    MPI_Datatype second = pool.register_type<MyStruct>(dt2);

    EXPECT_EQ(first, second);
}

// Regression: find<T>() for a T with NO mpi_type_traits<T> specialization at all must not hard-error.
// (find<T>()'s builtin/no-commit check has to short-circuit via nested if-constexpr, not a combined
// `has_static_type_v<T> && !mpi_type_traits<T>::has_to_be_committed` — the latter still requires
// mpi_type_traits<T>::has_to_be_committed to be a well-formed expression even when the left side is
// false, which it is not for a genuinely trait-less T.)
TEST(TypePoolValueTypeTest, FindOnTraitlessTypeDoesNotHardError) {
    kamping::v2::type_pool pool;
    EXPECT_FALSE(pool.find<TraitlessStruct>().has_value());

    MPI_Datatype dt = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(static_cast<int>(sizeof(TraitlessStruct)), MPI_BYTE, &dt);
    MPI_Datatype registered = pool.register_type<TraitlessStruct>(dt);
    ASSERT_NE(registered, MPI_DATATYPE_NULL);

    auto found = pool.find<TraitlessStruct>();
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, registered);
}

// ── with_value_pool / with_auto_value_pool: resolve the payload, not the pair ──────────────

TEST(TypePoolValueTypeTest, WithAutoValuePoolResolvesPayloadOnly) {
    kamping::v2::type_pool                pool;
    std::vector<std::pair<MyStruct, int>> pairs{{{1, 2.0}, 0}};

    auto view = pairs | views::with_auto_value_pool(pool);
    ASSERT_TRUE(pool.find<MyStruct>().has_value());
    EXPECT_EQ(kamping::v2::value_type(view), *pool.find<MyStruct>());
}

TEST(TypePoolValueTypeTest, WithValuePoolUsesOneShotRegisteredType) {
    kamping::v2::type_pool pool;
    MPI_Datatype           dt = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(static_cast<int>(sizeof(MyStruct)), MPI_BYTE, &dt);
    MPI_Datatype registered = pool.register_type<MyStruct>(dt);

    std::vector<std::pair<MyStruct, int>> pairs{{{1, 2.0}, 0}};
    auto                                  view = pairs | views::with_value_pool(pool);
    EXPECT_EQ(kamping::v2::value_type(view), registered);
}

// ── end-to-end: with_value_type/with_auto_value_pool + flatten_v() through a real alltoallv ────
//
// Exercises the composition sparse_alltoall.md's own transform_messages(with_auto_pool(...)) idea
// is modeled on: a structured (sparse) send buffer carrying a non-builtin payload, annotated on the
// value-type channel, flattened in-pipeline, and actually sent/received across ranks — not just
// type-checked in isolation. If flatten_v_view's mpi_type() override (honoring Source's
// mpi_value_type() instead of the deduced fallback) were wrong, or if the wrong MPI type were used
// on the wire, the received struct fields would come back corrupted or MPI would reject the
// mismatched type signature outright.
TEST(ValueTypeEndToEndTest, SparseFlattenWithValuePoolThroughAlltoallv) {
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    kamping::v2::type_pool pool;

    // rank r sends (r+1) copies of MyStruct{r*10+j, r+j} to rank j, as sparse (destination, buffer)
    // pairs in reverse rank order (exercises out-of-order layout, matching the existing
    // SparseFlattenSendBuffer / infer()-driven set_comm_size test for the plain-int case).
    std::vector<std::pair<int, std::vector<MyStruct>>> per_dest;
    for (int j = size - 1; j >= 0; --j) {
        std::vector<MyStruct> msgs;
        for (int k = 0; k < rank + 1; ++k) {
            msgs.push_back(MyStruct{rank * 10 + j, static_cast<double>(rank + j)});
        }
        per_dest.emplace_back(j, std::move(msgs));
    }

    std::vector<MyStruct> recv_data;
    kamping::v2::alltoallv(
        per_dest | views::with_auto_value_pool(pool) | views::flatten_v(),
        recv_data | views::with_type(pool.register_type<MyStruct>()) | views::auto_recv_v
    );

    std::vector<MyStruct> expected;
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < i + 1; ++k) {
            expected.push_back(MyStruct{i * 10 + rank, static_cast<double>(i + rank)});
        }
    }
    ASSERT_EQ(recv_data.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(recv_data[i].x, expected[i].x);
        EXPECT_DOUBLE_EQ(recv_data[i].y, expected[i].y);
    }
}

// Regression: a value_destination_pair source whose payload is itself an (integral, integral)
// pair -- e.g. std::pair<std::int64_t, std::int64_t>, the shape a reduce-by-key key/value
// contribution naturally takes -- must survive flatten_v() | with_auto_pool() intact. Once
// flattened, the flat buffer's own element type (the pair) structurally *also* matches
// value_destination_pair (its second field satisfies the `rank` concept just by being an
// integer). with_pool/with_auto_pool must therefore resolve via flat_element_t (always "this
// buffer's own element type"), not payload_element_t (which would reinterpret the already-flat
// buffer as another layer of (value, destination) pairs and truncate the payload to just the
// pair's first field, corrupting every element on the wire) -- see flat_element_t's doc comment
// in payload.hpp.
TEST(ValueTypeEndToEndTest, FlattenedIntPairPayloadSurvivesWithAutoPool) {
    using Pair = std::pair<std::int64_t, std::int64_t>;

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    kamping::v2::type_pool pool;

    // Every rank sends one (key, value) = (rank, rank * 100 + j) pair to every rank j.
    std::vector<std::pair<Pair, int>> msgs;
    for (int j = 0; j < size; ++j) {
        msgs.emplace_back(Pair{rank, rank * 100 + j}, j);
    }

    static_assert(std::is_same_v<kamping::v2::payload_element_t<decltype(msgs)>, Pair>);
    auto flattened = msgs | views::flatten_v();
    // with_auto_pool (the plain buffer-type channel) resolves via flat_element_t, not
    // payload_element_t -- see flat_element_t's doc comment for why the two must differ here.
    static_assert(std::is_same_v<kamping::v2::flat_element_t<decltype(flattened)>, Pair>);

    std::vector<Pair> recv_data;
    kamping::v2::alltoallv(
        flattened | views::with_auto_pool(pool),
        recv_data | views::with_auto_pool(pool) | views::auto_recv_v
    );

    std::vector<Pair> expected;
    for (int i = 0; i < size; ++i) {
        expected.emplace_back(i, i * 100 + rank);
    }
    std::ranges::sort(recv_data);
    std::ranges::sort(expected);
    EXPECT_EQ(recv_data, expected);
}
