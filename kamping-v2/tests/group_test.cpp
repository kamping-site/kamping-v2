// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <numeric>

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/comm.hpp"
#include "mpi/group.hpp"
#include "mpi/handle.hpp"

using namespace mpi::experimental;

// ── Concept checks ────────────────────────────────────────────────────────────

static_assert(convertible_to_mpi_handle<group_view, MPI_Group>);
static_assert(convertible_to_mpi_handle<group, MPI_Group>);
static_assert(convertible_to_mpi_handle<comm_view, MPI_Comm>);
static_assert(convertible_to_mpi_handle<comm, MPI_Comm>);

// ── group_view ────────────────────────────────────────────────────────────────

TEST(GroupViewTest, WrapsHandle) {
    // MPI_COMM_WORLD group as a raw handle for view testing
    MPI_Group world_group = MPI_GROUP_EMPTY;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    group_view gv(world_group);
    EXPECT_EQ(gv.mpi_handle(), world_group);

    MPI_Group_free(&world_group);
}

// ── group ─────────────────────────────────────────────────────────────────────

TEST(GroupTest, EmptyGroup) {
    group g = group::empty();
    EXPECT_EQ(g.mpi_handle(), MPI_GROUP_EMPTY);
}

TEST(GroupTest, FromNative) {
    MPI_Group raw = MPI_GROUP_EMPTY;
    MPI_Comm_group(MPI_COMM_WORLD, &raw);
    EXPECT_NE(raw, MPI_GROUP_EMPTY);

    group g = group::from_native(raw);
    EXPECT_EQ(g.mpi_handle(), raw);
    // g owns raw; it will be freed by g's destructor.
}

TEST(GroupTest, MoveConstructTransfersOwnership) {
    MPI_Group raw = MPI_GROUP_EMPTY;
    MPI_Comm_group(MPI_COMM_WORLD, &raw);

    group a = group::from_native(raw);
    group b = std::move(a);

    EXPECT_EQ(a.mpi_handle(), MPI_GROUP_EMPTY); // moved-from is MPI_GROUP_EMPTY
    EXPECT_EQ(b.mpi_handle(), raw);
}

TEST(GroupTest, MoveAssignTransfersOwnership) {
    MPI_Group raw_a = MPI_GROUP_EMPTY;
    MPI_Group raw_b = MPI_GROUP_EMPTY;
    MPI_Comm_group(MPI_COMM_WORLD, &raw_a);
    MPI_Comm_group(MPI_COMM_WORLD, &raw_b);

    group a = group::from_native(raw_a);
    group b = group::from_native(raw_b);
    b       = std::move(a);

    EXPECT_EQ(a.mpi_handle(), MPI_GROUP_EMPTY);
    EXPECT_EQ(b.mpi_handle(), raw_a);
    // raw_b was freed by b's move assignment.
}

TEST(GroupTest, ImplicitConversionToView) {
    MPI_Group raw = MPI_GROUP_EMPTY;
    MPI_Comm_group(MPI_COMM_WORLD, &raw);
    group      g  = group::from_native(raw);
    group_view gv = g; // implicit conversion
    EXPECT_EQ(gv.mpi_handle(), raw);
}

// ── group_accessors ───────────────────────────────────────────────────────────

TEST(GroupAccessorsTest, SizeMatchesCommSize) {
    comm_view world(MPI_COMM_WORLD);
    group     g = world.group();
    EXPECT_EQ(g.size(), world.size());
}

TEST(GroupAccessorsTest, RankMatchesCommRank) {
    comm_view world(MPI_COMM_WORLD);
    group     g   = world.group();
    auto      rnk = g.rank();
    ASSERT_TRUE(rnk.has_value());
    EXPECT_EQ(*rnk, world.rank());
}

TEST(GroupAccessorsTest, ContainsSelfTrue) {
    comm_view world(MPI_COMM_WORLD);
    group     g = world.group();
    EXPECT_TRUE(g.contains_self());
}

TEST(GroupAccessorsTest, CompareWorldWithItself) {
    comm_view world(MPI_COMM_WORLD);
    group     a = world.group();
    group     b = world.group();
    EXPECT_EQ(a.compare(b), GroupEquality::Identical);
}

// ── comm (owning) ─────────────────────────────────────────────────────────────

TEST(CommTest, ConceptSatisfied) {
    static_assert(convertible_to_mpi_handle<comm, MPI_Comm>);
}

TEST(CommTest, DupPreservesRankAndSize) {
    comm_view world(MPI_COMM_WORLD);

    // Free function: default-constructed comm as out-parameter
    comm dup_comm;
    dup(world, dup_comm);
    EXPECT_EQ(dup_comm.rank(), world.rank());
    EXPECT_EQ(dup_comm.size(), world.size());
    EXPECT_NE(dup_comm.mpi_handle(), MPI_COMM_NULL);
    EXPECT_NE(dup_comm.mpi_handle(), MPI_COMM_WORLD);

    // Member dup() on an owning comm
    comm dup2 = dup_comm.dup();
    EXPECT_EQ(dup2.rank(), world.rank());
    EXPECT_EQ(dup2.size(), world.size());
}

TEST(CommTest, MoveConstructTransfersOwnership) {
    comm_view world(MPI_COMM_WORLD);
    comm      a;
    dup(world, a);
    MPI_Comm raw = a.mpi_handle();
    comm     b   = std::move(a);

    EXPECT_EQ(a.mpi_handle(), MPI_COMM_NULL);
    EXPECT_EQ(b.mpi_handle(), raw);
}

TEST(CommTest, SplitEvenOdd) {
    comm_view world(MPI_COMM_WORLD);
    int       my_rank = world.rank();
    int       color   = my_rank % 2; // 0 = even, 1 = odd
    comm      sub;
    split(world, color, my_rank, sub);

    int world_size = world.size();
    int even_size  = (world_size + 1) / 2;
    int odd_size   = world_size / 2;
    int expected   = (color == 0) ? even_size : odd_size;

    EXPECT_EQ(sub.size(), expected);
    EXPECT_NE(sub.mpi_handle(), MPI_COMM_NULL);
}

TEST(CommTest, GroupFromCommView) {
    comm_view world(MPI_COMM_WORLD);
    group     g = world.group();
    EXPECT_EQ(g.size(), world.size());
}

#if defined(MPI_VERSION) && MPI_VERSION >= 4

TEST(CommTest, CommFromGroup) {
    comm_view world(MPI_COMM_WORLD);
    group     g        = world.group();
    comm      from_grp = comm::from_group(g, "test-tag");

    EXPECT_EQ(from_grp.rank(), world.rank());
    EXPECT_EQ(from_grp.size(), world.size());
}

TEST(CommTest, FromGroupWithGroupView) {
    comm_view  world(MPI_COMM_WORLD);
    group      g  = world.group();
    group_view gv = g;
    comm       c  = comm::from_group(gv, "view-tag");
    EXPECT_EQ(c.size(), world.size());
}

TEST(CommTest, FromGroupWithRawHandle) {
    MPI_Group raw_g = MPI_GROUP_EMPTY;
    MPI_Comm_group(MPI_COMM_WORLD, &raw_g);
    comm c = comm::from_group(raw_g, "raw-tag");
    EXPECT_EQ(c.size(), static_cast<int>(comm_view(MPI_COMM_WORLD).size()));
    MPI_Group_free(&raw_g);
}

#endif

TEST(CommTest, ReleaseRelinquishesOwnership) {
    comm_view world(MPI_COMM_WORLD);
    comm      a;
    dup(world, a);
    MPI_Comm raw      = a.mpi_handle();
    MPI_Comm released = a.release();
    EXPECT_EQ(released, raw);
    EXPECT_EQ(a.mpi_handle(), MPI_COMM_NULL);
    MPI_Comm_free(&released);
}

TEST(CommViewTest, DupViaAccessorBase) {
    comm_view world(MPI_COMM_WORLD);
    comm      d = world.dup();
    EXPECT_EQ(d.rank(), world.rank());
    EXPECT_EQ(d.size(), world.size());
    EXPECT_NE(d.mpi_handle(), MPI_COMM_WORLD);
}

TEST(CommViewTest, SplitViaAccessorBase) {
    comm_view world(MPI_COMM_WORLD);
    int       color = world.rank() % 2;
    comm      sub   = world.split(color, world.rank());
    EXPECT_NE(sub.mpi_handle(), MPI_COMM_NULL);
}

TEST(CommTest, FromNative) {
    MPI_Comm raw = MPI_COMM_NULL;
    MPI_Comm_dup(MPI_COMM_WORLD, &raw);
    comm c = comm::from_native(raw);
    EXPECT_EQ(c.mpi_handle(), raw);
    // freed by c's destructor
}

// ── group new tests ───────────────────────────────────────────────────────────

TEST(GroupTest, ReleaseRelinquishesOwnership) {
    MPI_Group raw = MPI_GROUP_EMPTY;
    MPI_Comm_group(MPI_COMM_WORLD, &raw);
    group     g        = group::from_native(raw);
    MPI_Group released = g.release();
    EXPECT_EQ(released, raw);
    EXPECT_EQ(g.mpi_handle(), MPI_GROUP_EMPTY);
    MPI_Group_free(&released);
}

// ── Rank translation ──────────────────────────────────────────────────────────

TEST(GroupTranslateTest, TranslateRankSelf) {
    comm_view world(MPI_COMM_WORLD);
    group     g = world.group();
    // Every rank in the world group translates to itself.
    auto result = g.translate_rank(world.rank(), g);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, world.rank());
}

TEST(GroupTranslateTest, TranslateRankNotInOther) {
    comm_view world(MPI_COMM_WORLD);
    if (world.size() < 2) {
        GTEST_SKIP() << "need at least 2 ranks";
    }
    group world_g = world.group();
    // Build a subgroup containing only rank 0; then ask where rank 1 maps.
    group sub = world_g.include(std::array{0});
    // rank 1 of world_g is not in sub → MPI_UNDEFINED → nullopt
    auto result = world_g.translate_rank(1, sub);
    EXPECT_FALSE(result.has_value());
}

TEST(GroupTranslateTest, TranslateRanksAllPresent) {
    comm_view world(MPI_COMM_WORLD);
    group     g   = world.group();
    auto      sz  = static_cast<size_t>(world.size());
    std::vector<int> all_ranks(sz);
    std::iota(all_ranks.begin(), all_ranks.end(), 0);

    auto translated = g.translate_ranks(all_ranks, g);
    ASSERT_EQ(static_cast<int>(translated.size()), sz);
    for (size_t i = 0; i < sz; ++i) {
        ASSERT_TRUE(translated[i].has_value());
        EXPECT_EQ(*translated[i], i);
    }
}

TEST(GroupTranslateTest, TranslateRanksMixed) {
    comm_view world(MPI_COMM_WORLD);
    if (world.size() < 2) {
        GTEST_SKIP() << "need at least 2 ranks";
    }
    group world_g = world.group();
    // sub contains only rank 0
    group            sub      = world_g.include(std::array{0});
    std::vector<int> to_xlate = {0, 1};
    auto             result   = world_g.translate_ranks(to_xlate, sub);
    ASSERT_EQ(result.size(), 2u);
    ASSERT_TRUE(result[0].has_value());
    EXPECT_EQ(*result[0], 0); // rank 0 maps to rank 0 in sub
    EXPECT_FALSE(result[1].has_value()); // rank 1 not in sub
}

// ── Subgroup selection ────────────────────────────────────────────────────────

TEST(GroupIncludeTest, IncludeSubsetPreservesOrder) {
    comm_view world(MPI_COMM_WORLD);
    if (world.size() < 2) {
        GTEST_SKIP() << "need at least 2 ranks";
    }
    group world_g = world.group();
    group sub     = world_g.include(std::array{0});
    EXPECT_EQ(sub.size(), 1);
}

TEST(GroupIncludeTest, IncludeAllRanks) {
    comm_view        world(MPI_COMM_WORLD);
    group            world_g = world.group();
    auto             sz      = static_cast<size_t>(world.size());
    std::vector<int> all(sz);
    std::iota(all.begin(), all.end(), 0);
    group full = world_g.include(all);
    EXPECT_EQ(full.size(), sz);
    EXPECT_EQ(full.compare(world_g), GroupEquality::Identical);
}

TEST(GroupExcludeTest, ExcludeOneRank) {
    comm_view world(MPI_COMM_WORLD);
    if (world.size() < 2) {
        GTEST_SKIP() << "need at least 2 ranks";
    }
    group world_g = world.group();
    group sub     = world_g.exclude(std::array{0});
    EXPECT_EQ(sub.size(), world.size() - 1);
}

TEST(GroupIncludeRangesTest, IncludeRangeAllRanks) {
    comm_view                      world(MPI_COMM_WORLD);
    group                          world_g = world.group();
    std::vector<std::array<int, 3>> all_range{{0, world.size() - 1, 1}};
    group                          ranged = world_g.include_ranges(all_range);
    EXPECT_EQ(ranged.size(), world.size());
    EXPECT_EQ(ranged.compare(world_g), GroupEquality::Identical);
}

TEST(GroupIncludeRangesTest, IncludeEvenRanks) {
    comm_view world(MPI_COMM_WORLD);
    if (world.size() < 2) {
        GTEST_SKIP() << "need at least 2 ranks";
    }
    group                          world_g    = world.group();
    int                            last_even  = ((world.size() - 1) / 2) * 2;
    int                            even_count = last_even / 2 + 1;
    std::vector<std::array<int, 3>> even_range{{0, last_even, 2}};
    group                          evens = world_g.include_ranges(even_range);
    EXPECT_EQ(evens.size(), even_count);
}

TEST(GroupExcludeRangesTest, ExcludeRangeAll) {
    comm_view                      world(MPI_COMM_WORLD);
    group                          world_g = world.group();
    std::vector<std::array<int, 3>> all_range{{0, world.size() - 1, 1}};
    group                          empty_g = world_g.exclude_ranges(all_range);
    EXPECT_EQ(empty_g.size(), 0);
}

// ── Set algebra ───────────────────────────────────────────────────────────────

TEST(GroupSetAlgebraTest, UnionWithSelf) {
    comm_view world(MPI_COMM_WORLD);
    group     a = world.group();
    group     b = world.group();
    group     u = set_union(a, b);
    EXPECT_EQ(u.size(), world.size());
    EXPECT_EQ(u.compare(a), GroupEquality::Identical);
}

TEST(GroupSetAlgebraTest, IntersectionWithSelf) {
    comm_view world(MPI_COMM_WORLD);
    group     a    = world.group();
    group     b    = world.group();
    group     isec = intersection(a, b);
    EXPECT_EQ(isec.size(), world.size());
    EXPECT_EQ(isec.compare(a), GroupEquality::Identical);
}

TEST(GroupSetAlgebraTest, DifferenceWithSelf) {
    comm_view world(MPI_COMM_WORLD);
    group     a    = world.group();
    group     b    = world.group();
    group     diff = difference(a, b);
    EXPECT_EQ(diff.size(), 0);
}

TEST(GroupSetAlgebraTest, IntersectionDisjoint) {
    comm_view world(MPI_COMM_WORLD);
    if (world.size() < 2) {
        GTEST_SKIP() << "need at least 2 ranks";
    }
    group world_g = world.group();
    group sub0    = world_g.include(std::array{0});
    group sub1    = world_g.include(std::array{1});
    group isec    = intersection(sub0, sub1);
    EXPECT_EQ(isec.size(), 0);
}

TEST(GroupSetAlgebraTest, UnionDisjointSubgroups) {
    comm_view world(MPI_COMM_WORLD);
    if (world.size() < 2) {
        GTEST_SKIP() << "need at least 2 ranks";
    }
    group world_g = world.group();
    group sub0    = world_g.include(std::array{0});
    group sub1    = world_g.include(std::array{1});
    group u       = set_union(sub0, sub1);
    EXPECT_EQ(u.size(), 2);
}

TEST(GroupSetAlgebraTest, DifferenceRemovesSubset) {
    comm_view world(MPI_COMM_WORLD);
    if (world.size() < 2) {
        GTEST_SKIP() << "need at least 2 ranks";
    }
    group world_g = world.group();
    group sub0    = world_g.include(std::array{0});
    group diff    = difference(world_g, sub0);
    EXPECT_EQ(diff.size(), world.size() - 1);
}
