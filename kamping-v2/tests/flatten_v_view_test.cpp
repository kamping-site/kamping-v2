// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <cstddef>
#include <ranges>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kamping/v2/views.hpp"
#include "mpi/buffer.hpp"

using namespace ::testing;

namespace {

// Materializes a flattened view into (data, counts, displs) using the public buffer
// protocol accessors. Triggers the lazy flatten on first access.
template <typename View>
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> materialize(View const& view) {
    auto const  n = static_cast<std::ptrdiff_t>(mpi::experimental::count(view));
    auto const* p = mpi::experimental::ptr(view);
    auto const  c = mpi::experimental::counts(view);
    auto const  d = mpi::experimental::displs(view);
    return {
        std::vector<int>(p, p + n),
        std::vector<int>(std::ranges::begin(c), std::ranges::end(c)),
        std::vector<int>(std::ranges::begin(d), std::ranges::end(d)),
    };
}

} // namespace

// ── Dense range-of-ranges ───────────────────────────────────────────────────────────
// One inner range per rank; rank count is the source size, no set_comm_size needed.
TEST(FlattenVTest, DenseNested) {
    std::vector<std::vector<int>> per_rank = {{10, 11}, {}, {30, 31, 32}};

    auto const [data, counts, displs] = materialize(per_rank | kamping::v2::views::flatten_v());

    EXPECT_THAT(data, ElementsAre(10, 11, 30, 31, 32));
    EXPECT_THAT(counts, ElementsAre(2, 0, 3));
    EXPECT_THAT(displs, ElementsAre(0, 2, 2));
}

TEST(FlattenVTest, DenseEmpty) {
    std::vector<std::vector<int>> per_rank = {{}, {}};

    auto const [data, counts, displs] = materialize(per_rank | kamping::v2::views::flatten_v());

    EXPECT_THAT(data, IsEmpty());
    EXPECT_THAT(counts, ElementsAre(0, 0));
    EXPECT_THAT(displs, ElementsAre(0, 0));
}

// User-provided (borrowed) flat buffer, counts, and displs — pre-sized, no resize. The
// view writes through the borrows, so the caller's own containers are populated.
TEST(FlattenVTest, UserProvidedBuffers) {
    std::vector<std::vector<int>> per_rank = {{1, 2}, {3}};
    std::vector<int>              flat_buf(3);
    std::vector<int>              counts(2);
    std::vector<int>              displs(2);

    auto view = per_rank | kamping::v2::views::flatten_v(flat_buf, counts, displs);
    (void)materialize(view); // triggers the flatten through the borrowed buffers

    EXPECT_THAT(flat_buf, ElementsAre(1, 2, 3));
    EXPECT_THAT(counts, ElementsAre(2, 1));
    EXPECT_THAT(displs, ElementsAre(0, 2));
}

// ── Sparse (destination, buffer) pairs ──────────────────────────────────────────────
// Ranks appear out of order, repeated (rank 2), and absent (rank 1). The communicator
// size must be supplied via set_comm_size() before the layout can be computed.
TEST(FlattenVTest, SparseOutOfOrderRepeatedAbsent) {
    std::vector<std::pair<int, std::vector<int>>> per_rank = {
        {2, {20, 21}},
        {0, {0}},
        {2, {22}},
        {3, {30, 31, 32}},
    };

    auto view = per_rank | kamping::v2::views::flatten_v();
    view.set_comm_size(4);
    auto const [data, counts, displs] = materialize(view);

    // rank0: [0], rank1: [], rank2: [20,21,22] (two contributions merged), rank3: [30,31,32]
    EXPECT_THAT(counts, ElementsAre(1, 0, 3, 3));
    EXPECT_THAT(displs, ElementsAre(0, 1, 1, 4));
    EXPECT_THAT(data, ElementsAre(0, 20, 21, 22, 30, 31, 32));
}

TEST(FlattenVTest, SparseNoPairsAllRanksZero) {
    std::vector<std::pair<int, std::vector<int>>> per_rank;

    auto view = per_rank | kamping::v2::views::flatten_v();
    view.set_comm_size(3);
    auto const [data, counts, displs] = materialize(view);

    EXPECT_THAT(data, IsEmpty());
    EXPECT_THAT(counts, ElementsAre(0, 0, 0));
    EXPECT_THAT(displs, ElementsAre(0, 0, 0));
}

// An unordered_map<rank, buffer> is a forward range of (const rank, buffer) pairs, so it
// flows through the sparse path directly. Each rank appears once; the per-rank cursors are
// seeded from the displacements, so the flattened layout is determined by rank index and is
// independent of the map's iteration order.
TEST(FlattenVTest, SparseUnorderedMap) {
    std::unordered_map<int, std::vector<int>> per_rank = {
        {0, {0}},
        {2, {20, 21}},
        {3, {30, 31, 32}},
    };

    auto view = per_rank | kamping::v2::views::flatten_v();
    view.set_comm_size(4);
    auto const [data, counts, displs] = materialize(view);

    EXPECT_THAT(counts, ElementsAre(1, 0, 2, 3));
    EXPECT_THAT(displs, ElementsAre(0, 1, 1, 3));
    EXPECT_THAT(data, ElementsAre(0, 20, 21, 30, 31, 32));
}

// ── (value, destination) pairs ──────────────────────────────────────────────────────
// Each pair contributes a single value to its destination rank.
TEST(FlattenVTest, ValueDestination) {
    std::vector<std::pair<int, int>> items = {
        {100, 1},
        {200, 0},
        {101, 1},
        {300, 2},
    };

    auto view = items | kamping::v2::views::flatten_v();
    view.set_comm_size(3);
    auto const [data, counts, displs] = materialize(view);

    // rank0: [200], rank1: [100,101] (input order preserved), rank2: [300]
    EXPECT_THAT(counts, ElementsAre(1, 2, 1));
    EXPECT_THAT(displs, ElementsAre(0, 1, 3));
    EXPECT_THAT(data, ElementsAre(200, 100, 101, 300));
}

// Projection-based flattening needs no dedicated API: the caller derives the destination
// from each element with a plain std::ranges adaptor, producing a (value, rank) pair range
// that flows through the value-destination path. Here the rank is value % comm_size.
TEST(FlattenVTest, ProjectionViaStdRanges) {
    struct Item {
        int payload;
        int owner;
    };
    std::vector<Item> items = {{10, 1}, {20, 0}, {11, 1}, {30, 2}};

    auto pairs = items | std::views::transform([](Item const& it) { return std::pair{it.payload, it.owner}; });

    auto view = pairs | kamping::v2::views::flatten_v();
    view.set_comm_size(3);
    auto const [data, counts, displs] = materialize(view);

    // rank0: [20], rank1: [10,11], rank2: [30]
    EXPECT_THAT(counts, ElementsAre(1, 2, 1));
    EXPECT_THAT(displs, ElementsAre(0, 1, 3));
    EXPECT_THAT(data, ElementsAre(20, 10, 11, 30));
}

// Lazy re-flatten: changing the comm size invalidates the cached layout.
TEST(FlattenVTest, ReflattensAfterSetCommSize) {
    std::vector<std::pair<int, int>> items = {{7, 0}, {8, 2}};

    auto view = items | kamping::v2::views::flatten_v();

    view.set_comm_size(3);
    EXPECT_THAT(std::get<1>(materialize(view)), ElementsAre(1, 0, 1));

    view.set_comm_size(4);
    EXPECT_THAT(std::get<1>(materialize(view)), ElementsAre(1, 0, 1, 0));
}
