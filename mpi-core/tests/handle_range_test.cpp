// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <array>
#include <iterator>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/handle.hpp"

namespace me = mpi::experimental;

namespace {

/// Tier 2: exposes its storage as a view returned by value.
struct span_member {
    std::vector<MPI_Request> requests;
    std::span<MPI_Request>   mpi_handle_range() {
        return requests;
    }
};

/// Tier 2: exposes its storage by reference.
struct ref_member {
    std::vector<MPI_Request>  requests;
    std::vector<MPI_Request>& mpi_handle_range() {
        return requests;
    }
};

/// Tier 2, but invalid: returns a copy, which MPI would write completed handles into.
struct copy_member {
    std::vector<MPI_Request> requests;
    std::vector<MPI_Request> mpi_handle_range() {
        return requests;
    }
};

/// Itself a contiguous handle range, but the member must take priority over the passthrough.
struct range_with_member : std::vector<MPI_Request> {
    std::vector<MPI_Request> other;
    std::span<MPI_Request>   mpi_handle_range() {
        return other;
    }
};

/// Has a member, but the trait specialization below must take priority.
struct traits_and_member {
    std::vector<MPI_Request> from_traits;
    std::vector<MPI_Request> from_member;
    std::span<MPI_Request>   mpi_handle_range() {
        return from_member;
    }
};

/// Third-party type adapted only via handle_range_traits.
struct third_party {
    MPI_Request storage[3];
};

template <typename T>
concept accepts_handle_range = requires(T&& t) { me::handle_range(std::forward<T>(t)); };

} // namespace

template <>
struct mpi::experimental::handle_range_traits<traits_and_member> {
    static std::span<MPI_Request> handle_range(traits_and_member& t) {
        return t.from_traits;
    }
};

template <>
struct mpi::experimental::handle_range_traits<third_party> {
    static std::span<MPI_Request> handle_range(third_party& t) {
        return t.storage;
    }
};

// ── concept ──────────────────────────────────────────────────────────────────

static_assert(me::convertible_to_mpi_handle_range<std::vector<MPI_Request>, MPI_Request>);
static_assert(me::convertible_to_mpi_handle_range<std::array<MPI_Request, 4>, MPI_Request>);
static_assert(me::convertible_to_mpi_handle_range<MPI_Request[4], MPI_Request>);
static_assert(me::convertible_to_mpi_handle_range<std::span<MPI_Request>, MPI_Request>);
static_assert(me::convertible_to_mpi_handle_range<std::vector<MPI_Status>, MPI_Status>);
static_assert(me::convertible_to_mpi_handle_range<span_member, MPI_Request>);
static_assert(me::convertible_to_mpi_handle_range<ref_member, MPI_Request>);
static_assert(me::convertible_to_mpi_handle_range<third_party, MPI_Request>);

// Wrong handle type, non-handle elements, read-only storage.
static_assert(!me::convertible_to_mpi_handle_range<std::vector<MPI_Request>, MPI_Status>);
// (Not std::vector<int>: in MPICH and its derivatives, MPI_Request and the other handles are typedefs of int.)
static_assert(!me::convertible_to_mpi_handle_range<std::vector<double>, MPI_Request>);
static_assert(!me::convertible_to_mpi_handle_range<std::vector<MPI_Request> const, MPI_Request>);
static_assert(!me::convertible_to_mpi_handle_range<std::span<MPI_Request const>, MPI_Request>);
// A copy is not borrowed: MPI's write-back of MPI_REQUEST_NULL would be lost.
static_assert(!me::convertible_to_mpi_handle_range<copy_member, MPI_Request>);
// Not a range at all.
static_assert(!me::convertible_to_mpi_handle_range<MPI_Request, MPI_Request>);

// A handle range need not be sized (e.g. a pointer with an unreachable end); arguments that need a count require it.
using unbounded_requests = std::ranges::subrange<MPI_Request*, std::unreachable_sentinel_t>;
static_assert(me::convertible_to_mpi_handle_range<unbounded_requests, MPI_Request>);

// ── return types ─────────────────────────────────────────────────────────────

// The passthrough returns the argument by reference, never a copy, and rejects temporaries.
static_assert(
    std::is_same_v<decltype(me::handle_range(std::declval<std::vector<MPI_Request>&>())), std::vector<MPI_Request>&>
);
static_assert(accepts_handle_range<std::vector<MPI_Request>&>);
static_assert(!accepts_handle_range<std::vector<MPI_Request>>);
// Hooks keep exactly what they return: a view by value, a container by reference.
static_assert(std::is_same_v<decltype(me::handle_range(std::declval<span_member&>())), std::span<MPI_Request>>);
static_assert(std::is_same_v<decltype(me::handle_range(std::declval<ref_member&>())), std::vector<MPI_Request>&>);

// ── dispatch ─────────────────────────────────────────────────────────────────

TEST(HandleRangeTest, PassthroughRefersToStorage) {
    std::vector<MPI_Request> requests(3);
    auto&&                   range = me::handle_range(requests);
    EXPECT_EQ(&range, &requests);

    MPI_Request array[2];
    EXPECT_EQ(std::ranges::data(me::handle_range(array)), array);
}

TEST(HandleRangeTest, WritesReachStorage) {
    std::vector<MPI_Request> requests(2);
    for (auto& r: me::handle_range(requests)) {
        r = MPI_REQUEST_NULL;
    }
    EXPECT_EQ(requests[0], MPI_REQUEST_NULL);
    EXPECT_EQ(requests[1], MPI_REQUEST_NULL);

    span_member s{std::vector<MPI_Request>(2)};
    std::ranges::data(me::handle_range(s))[1] = MPI_REQUEST_NULL;
    EXPECT_EQ(s.requests[1], MPI_REQUEST_NULL);
}

TEST(HandleRangeTest, MemberRefersToStorage) {
    span_member s{std::vector<MPI_Request>(3)};
    EXPECT_EQ(std::ranges::data(me::handle_range(s)), s.requests.data());
    EXPECT_EQ(std::ranges::size(me::handle_range(s)), 3u);

    ref_member r{std::vector<MPI_Request>(2)};
    EXPECT_EQ(&me::handle_range(r), &r.requests);
}

TEST(HandleRangeTest, MemberTakesPriorityOverPassthrough) {
    range_with_member r;
    r.resize(1);
    r.other.resize(4);
    EXPECT_EQ(std::ranges::data(me::handle_range(r)), r.other.data());
    EXPECT_EQ(std::ranges::size(me::handle_range(r)), 4u);
}

TEST(HandleRangeTest, TraitsTakePriorityOverMember) {
    traits_and_member t{std::vector<MPI_Request>(2), std::vector<MPI_Request>(5)};
    EXPECT_EQ(std::ranges::data(me::handle_range(t)), t.from_traits.data());
    EXPECT_EQ(std::ranges::size(me::handle_range(t)), 2u);
}

TEST(HandleRangeTest, TraitsAdaptThirdPartyType) {
    third_party t{};
    EXPECT_EQ(std::ranges::data(me::handle_range(t)), t.storage);
    EXPECT_EQ(std::ranges::size(me::handle_range(t)), 3u);
}
