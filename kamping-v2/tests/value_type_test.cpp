// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/types/contiguous_type.hpp"
#include "kamping/types/mpi_type_traits.hpp"
#include "kamping/v2/type_pool.hpp"
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
