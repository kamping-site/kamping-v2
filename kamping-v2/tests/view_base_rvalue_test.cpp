// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <concepts>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/auto_counts_view.hpp"
#include "kamping/v2/views/auto_displs_view.hpp"
#include "kamping/v2/views/resize_v_view.hpp"
#include "kamping/v2/views/resize_view.hpp"
#include "kamping/v2/views/with_count_view.hpp"
#include "kamping/v2/views/with_counts_view.hpp"
#include "kamping/v2/views/with_displs_view.hpp"
#include "kamping/v2/views/with_type_view.hpp"
#include "kamping/v2/views/with_value_type_view.hpp"

using kamping::v2::auto_counts_view;
using kamping::v2::auto_displs_view;
using kamping::v2::owning_view;
using kamping::v2::ref_view;
using kamping::v2::resize_v_view;
using kamping::v2::resize_view;
using kamping::v2::with_count_view;
using kamping::v2::with_counts_view;
using kamping::v2::with_displs_view;
using kamping::v2::with_type_view;
using kamping::v2::with_value_type_view;

namespace {

std::vector<int> make_data() {
    return {1, 2, 3};
}

/// Checks that std::move(view).base() selects the && overload and moves the base out:
/// the returned Base&& must be usable to move-construct a (possibly move-only) Base, and the
/// view's underlying container must be left in the moved-from (empty) state.
template <typename View>
void expect_rvalue_base_moves_out(View view) {
    using Base = std::remove_cvref_t<decltype(view.base())>;
    static_assert(std::same_as<decltype(std::move(view).base()), Base&&>);

    Base extracted = std::move(view).base();
    EXPECT_EQ(extracted.underlying(), make_data());
    EXPECT_TRUE(view.underlying().empty());
}

} // namespace

// ── owning bases: std::move(view).base() must move, not copy ──────────────────
// Each Base below wraps an owning_view, which is move-only — before the && overload,
// std::move(view).base() bound to the const& overload and Base could not be constructed from it.

TEST(ViewBaseRvalueTest, ResizeView) {
    expect_rvalue_base_moves_out(resize_view(make_data()));
}

TEST(ViewBaseRvalueTest, WithTypeView) {
    expect_rvalue_base_moves_out(with_type_view(make_data(), MPI_INT));
}

TEST(ViewBaseRvalueTest, WithValueTypeView) {
    expect_rvalue_base_moves_out(with_value_type_view(make_data(), MPI_INT));
}

TEST(ViewBaseRvalueTest, WithCountView) {
    expect_rvalue_base_moves_out(with_count_view(make_data(), 3));
}

TEST(ViewBaseRvalueTest, WithCountsView) {
    expect_rvalue_base_moves_out(with_counts_view(make_data(), std::vector<int>{3}));
}

TEST(ViewBaseRvalueTest, WithDisplsView) {
    expect_rvalue_base_moves_out(with_displs_view(make_data(), std::vector<int>{0}));
}

TEST(ViewBaseRvalueTest, AutoCountsView) {
    expect_rvalue_base_moves_out(auto_counts_view(make_data(), std::vector<int>{3}));
}

TEST(ViewBaseRvalueTest, AutoDisplsView) {
    expect_rvalue_base_moves_out(
        auto_displs_view(with_counts_view(make_data(), std::vector<int>{3}), std::vector<int>(1))
    );
}

TEST(ViewBaseRvalueTest, ResizeVView) {
    expect_rvalue_base_moves_out(
        resize_v_view(with_displs_view(with_counts_view(make_data(), std::vector<int>{3}), std::vector<int>{0}))
    );
}

// ── borrowed base: the && overload hands out the ref_view, the data stays put ─

TEST(ViewBaseRvalueTest, BorrowedBaseLeavesDataUntouched) {
    std::vector<int> data = make_data();
    auto             view = resize_view(data);

    static_assert(std::same_as<decltype(std::move(view).base()), ref_view<std::vector<int>>&&>);
    ref_view<std::vector<int>> extracted = std::move(view).base();
    EXPECT_EQ(&extracted.base(), &data);
    EXPECT_EQ(data, make_data());
}
