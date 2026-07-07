// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <vector>

#include <gtest/gtest.h>

#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/resize_view.hpp"
#include "kamping/v2/views/with_type_view.hpp"

namespace views = kamping::v2::views;

using kamping::v2::owning_view;
using kamping::v2::ref_view;

// ── lvalue / const-lvalue overloads (existing behaviour) ──────────────────────

TEST(UnderlyingTest, LvalueReturnsNonConstRef) {
    std::vector<int> v{1, 2, 3};
    auto             view = ref_view<std::vector<int>>{v};

    static_assert(std::same_as<decltype(view.underlying()), std::vector<int>&>);
    EXPECT_EQ(&view.underlying(), &v);
}

TEST(UnderlyingTest, ConstLvalueReturnsConstRef) {
    std::vector<int> const v{1, 2, 3};
    auto const             view = ref_view<std::vector<int> const>{v};

    static_assert(std::same_as<decltype(view.underlying()), std::vector<int> const&>);
    EXPECT_EQ(&view.underlying(), &v);
}

// ── rvalue overload (the new behaviour) ───────────────────────────────────────

TEST(UnderlyingTest, RvalueOwningViewReturnsValue) {
    auto make_view = [] { return owning_view<std::vector<int>>{std::vector<int>{1, 2, 3}}; };

    // Return type must be by value, not a reference.
    static_assert(std::same_as<decltype(make_view().underlying()), std::vector<int>>);

    std::vector<int> result = make_view().underlying();
    EXPECT_EQ(result, (std::vector<int>{1, 2, 3}));
}

TEST(UnderlyingTest, RvalueOwningViewMoves) {
    // Verify zero-copy: the source vector is left empty after extraction.
    owning_view<std::vector<int>> view{std::vector<int>{4, 5, 6}};
    std::vector<int>              result = std::move(view).underlying();

    EXPECT_EQ(result, (std::vector<int>{4, 5, 6}));
    EXPECT_TRUE(view.underlying().empty()); // moved-from vector is empty
}

// ── rvalue extraction through a wrapping view layer ───────────────────────────

TEST(UnderlyingTest, RvalueChainedViewExtractsUnderlying) {
    // A resize_view wraps an owning_view<vector<int>> internally (via all()).
    // Calling underlying() && must peel through the resize_view layer and
    // move out from the owning_view.
    auto make_chained = [] {
        std::vector<int> v{7, 8, 9};
        return owning_view<std::vector<int>>{std::move(v)} | views::resize;
    };

    static_assert(std::same_as<decltype(make_chained().underlying()), std::vector<int>>);

    std::vector<int> result = make_chained().underlying();
    EXPECT_EQ(result, (std::vector<int>{7, 8, 9}));
}

TEST(UnderlyingTest, RvalueDoublyWrappedViewExtractsUnderlying) {
    // Two wrapping layers (resize then with_type).
    auto make_wrapped = [] {
        std::vector<int> v{10, 20, 30};
        return owning_view<std::vector<int>>{std::move(v)} | views::resize | views::with_type(MPI_INT);
    };

    static_assert(std::same_as<decltype(make_wrapped().underlying()), std::vector<int>>);

    std::vector<int> result = make_wrapped().underlying();
    EXPECT_EQ(result, (std::vector<int>{10, 20, 30}));
}
