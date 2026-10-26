// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <array>
#include <iterator>
#include <ranges>
#include <span>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/arrays.hpp"
#include "mpi/sentinels.hpp"

namespace me = mpi::experimental;

namespace {

/// Exposes its request storage as a view returned by value.
struct span_member {
    std::vector<MPI_Request> requests;
    std::span<MPI_Request>   mpi_handle_range() {
        return requests;
    }
};

using unbounded_requests = std::ranges::subrange<MPI_Request*, std::unreachable_sentinel_t>;

} // namespace

// The request array must be sized: its size is the request count.
static_assert(me::request_range<std::vector<MPI_Request>>);
static_assert(me::request_range<MPI_Request[4]>);
static_assert(me::request_range<span_member>);
static_assert(!me::request_range<unbounded_requests>);
static_assert(!me::request_range<std::vector<MPI_Status>>);

// Output arrays of handles go through handle_range; sized or not.
static_assert(me::output_array_of<std::vector<MPI_Status>, MPI_Status>);
static_assert(me::output_array_of<std::span<MPI_Status>, MPI_Status>);
static_assert(me::output_array_of<std::ranges::subrange<MPI_Status*, std::unreachable_sentinel_t>, MPI_Status>);
static_assert(!me::output_array_of<std::vector<MPI_Status> const, MPI_Status>);

// Output arrays of non-handles are plain contiguous ranges of exactly that element type.
static_assert(me::output_array_of<std::vector<int>, int>);
static_assert(me::output_array_of<std::vector<int>&, int>);
static_assert(me::output_array_of<int[4], int>);
static_assert(me::output_array_of<std::span<int>, int>);
static_assert(!me::output_array_of<std::vector<int> const, int>);
static_assert(!me::output_array_of<std::span<int const>, int>);
static_assert(!me::output_array_of<std::vector<long>, int>);

// Raw pointers carry no length and are never arrays, not even MPI_STATUSES_IGNORE.
static_assert(!me::output_array_or_ignore_of<MPI_Status*, MPI_Status>);
static_assert(!me::output_array_or_ignore_of<int*, int>);
// A single status is not an array of statuses.
static_assert(!me::output_array_or_ignore_of<MPI_Status, MPI_Status>);

// statuses_ignore may replace an ignorable statuses array, but nothing else.
static_assert(me::output_array_or_ignore_of<me::statuses_ignore_t, MPI_Status>);
static_assert(me::output_array_or_ignore_of<me::statuses_ignore_t const&, MPI_Status>);
static_assert(!me::output_array_of<me::statuses_ignore_t, MPI_Status>);
static_assert(!me::output_array_or_ignore_of<me::statuses_ignore_t, int>);
static_assert(!me::convertible_to_mpi_handle_ptr<me::statuses_ignore_t, MPI_Status>);

TEST(OutputArrayTest, PointerToPassToMpi) {
    EXPECT_EQ(me::detail::output_array_ptr<MPI_Status>(me::statuses_ignore, 3), MPI_STATUSES_IGNORE);

    std::vector<MPI_Status> statuses(3);
    EXPECT_EQ(me::detail::output_array_ptr<MPI_Status>(statuses, 3), statuses.data());

    std::array<int, 3> indices{};
    EXPECT_EQ(me::detail::output_array_ptr<int>(indices, 2), indices.data());

    MPI_Status                                                      raw[3];
    std::ranges::subrange<MPI_Status*, std::unreachable_sentinel_t> unbounded{raw, std::unreachable_sentinel};
    EXPECT_EQ(me::detail::output_array_ptr<MPI_Status>(unbounded, 3), raw);
}
