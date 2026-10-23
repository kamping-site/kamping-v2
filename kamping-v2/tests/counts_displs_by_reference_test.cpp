// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <ranges>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/views/with_type_view.hpp"
#include "mpi/buffer.hpp"

namespace views = kamping::v2::views;

// counts() / displs() must forward a container returned by reference as a reference. Returning a copy would make
// writes through counts() (as done by infer()) land in a temporary.

namespace {

/// Variadic buffer exposing counts/displs via member functions that return its own containers by reference.
struct MemberBuf {
    std::vector<int> data;
    std::vector<int> counts;
    std::vector<int> displs;

    auto begin() {
        return data.begin();
    }
    auto end() {
        return data.end();
    }
    auto begin() const {
        return data.begin();
    }
    auto end() const {
        return data.end();
    }

    std::vector<int>& mpi_counts() {
        return counts;
    }
    std::vector<int> const& mpi_counts() const {
        return counts;
    }
    std::vector<int> const& mpi_displs() const {
        return displs;
    }
};

/// Variadic buffer exposing counts/displs only via buffer_traits.
struct TraitsBuf {
    std::vector<int> counts;
    std::vector<int> displs;
};

} // namespace

template <>
struct mpi::experimental::buffer_traits<TraitsBuf> {
    static std::vector<int>& counts(TraitsBuf& b) {
        return b.counts;
    }
    static std::vector<int> const& counts(TraitsBuf const& b) {
        return b.counts;
    }
    static std::vector<int>& displs(TraitsBuf& b) {
        return b.displs;
    }
    static std::vector<int> const& displs(TraitsBuf const& b) {
        return b.displs;
    }
};

TEST(CountsDisplsByReferenceTest, MemberAccessorsReturnReferences) {
    MemberBuf buf{{}, {0, 0}, {0, 1}};

    static_assert(std::same_as<decltype(mpi::experimental::counts(buf)), std::vector<int>&>);
    static_assert(std::same_as<decltype(mpi::experimental::counts(std::as_const(buf))), std::vector<int> const&>);
    static_assert(std::same_as<decltype(mpi::experimental::displs(buf)), std::vector<int> const&>);
    static_assert(mpi::experimental::has_mpi_counts_mutable<MemberBuf>);

    std::ranges::data(mpi::experimental::counts(buf))[1] = 42;
    EXPECT_EQ(buf.counts[1], 42);
    EXPECT_EQ(&mpi::experimental::displs(buf), &buf.displs);
}

TEST(CountsDisplsByReferenceTest, TraitsAccessorsReturnReferences) {
    TraitsBuf buf{{0, 0}, {0, 1}};

    static_assert(std::same_as<decltype(mpi::experimental::counts(buf)), std::vector<int>&>);
    static_assert(std::same_as<decltype(mpi::experimental::displs(buf)), std::vector<int>&>);
    static_assert(std::same_as<decltype(mpi::experimental::displs(std::as_const(buf))), std::vector<int> const&>);
    static_assert(mpi::experimental::has_mpi_counts_mutable<TraitsBuf>);

    std::ranges::data(mpi::experimental::counts(buf))[0] = 7;
    EXPECT_EQ(buf.counts[0], 7);
    EXPECT_EQ(&mpi::experimental::displs(buf), &buf.displs);
}

TEST(CountsDisplsByReferenceTest, ViewForwardsReferences) {
    MemberBuf buf{{}, {0, 0}, {0, 1}};
    auto      view = buf | views::with_type(MPI_INT);

    static_assert(std::same_as<decltype(mpi::experimental::counts(view)), std::vector<int>&>);
    static_assert(std::same_as<decltype(mpi::experimental::displs(view)), std::vector<int> const&>);
    static_assert(mpi::experimental::has_mpi_counts_mutable<decltype(view)>);

    std::ranges::data(mpi::experimental::counts(view))[1] = 5;
    EXPECT_EQ(buf.counts[1], 5);
    EXPECT_EQ(&mpi::experimental::displs(view), &buf.displs);
}
