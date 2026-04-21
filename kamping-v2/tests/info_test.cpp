#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/handle.hpp"
#include "mpi/info.hpp"
#include "mpi/thread_level.hpp"

using namespace mpi::experimental;

// ── info_view ────────────────────────────────────────────────────────────────

TEST(InfoViewTest, ConceptSatisfied) {
    static_assert(convertible_to_mpi_handle<info_view, MPI_Info>);
}

TEST(InfoViewTest, NullInfo) {
    info_view iv(MPI_INFO_NULL);
    EXPECT_EQ(iv.mpi_handle(), MPI_INFO_NULL);
    EXPECT_EQ(iv.native(), MPI_INFO_NULL);
}

// ── info construction / move ─────────────────────────────────────────────────

TEST(InfoTest, ConceptSatisfied) {
    static_assert(convertible_to_mpi_handle<info, MPI_Info>);
}

TEST(InfoTest, DefaultConstructCreatesHandle) {
    info i;
    EXPECT_NE(i.mpi_handle(), MPI_INFO_NULL);
}

TEST(InfoTest, MoveConstructTransfersOwnership) {
    info a;
    MPI_Info raw = a.mpi_handle();
    info     b(std::move(a));
    EXPECT_EQ(a.mpi_handle(), MPI_INFO_NULL); // moved-from is null
    EXPECT_EQ(b.mpi_handle(), raw);
}

TEST(InfoTest, MoveAssignTransfersOwnership) {
    info a;
    info b;
    MPI_Info raw = a.mpi_handle();
    b            = std::move(a);
    EXPECT_EQ(a.mpi_handle(), MPI_INFO_NULL);
    EXPECT_EQ(b.mpi_handle(), raw);
}

TEST(InfoTest, DupCreatesIndependentCopy) {
    info a;
    a.set("k", "v");
    info b = a.dup();
    EXPECT_NE(b.mpi_handle(), a.mpi_handle());
    EXPECT_EQ(b.get("k"), std::optional<std::string>{"v"});

    // Mutating b does not affect a.
    b.set("k", "other");
    EXPECT_EQ(a.get("k"), std::optional<std::string>{"v"});
}

// ── get / set / contains / erase ─────────────────────────────────────────────

TEST(InfoTest, SetAndGetString) {
    info i;
    i.set("hello", "world");
    EXPECT_EQ(i.get("hello"), std::optional<std::string>{"world"});
}

TEST(InfoTest, GetAbsentKeyReturnsNullopt) {
    info i;
    EXPECT_EQ(i.get("missing"), std::nullopt);
}

TEST(InfoTest, Contains) {
    info i;
    EXPECT_FALSE(i.contains("k"));
    i.set("k", "v");
    EXPECT_TRUE(i.contains("k"));
}

TEST(InfoTest, OverwriteExistingKey) {
    info i;
    i.set("k", "first");
    i.set("k", "second");
    EXPECT_EQ(i.get("k"), std::optional<std::string>{"second"});
    EXPECT_EQ(i.nkeys(), 1);
}

TEST(InfoTest, EraseKey) {
    info i;
    i.set("k", "v");
    EXPECT_EQ(i.nkeys(), 1);
    i.erase("k");
    EXPECT_EQ(i.nkeys(), 0);
    EXPECT_EQ(i.get("k"), std::nullopt);
}

// ── info_value_traits: typed set / get ───────────────────────────────────────

TEST(InfoTest, SetGetBool) {
    info i;
    i.set("flag", true);
    EXPECT_EQ(i.get<bool>("flag"), std::optional<bool>{true});
    i.set("flag", false);
    EXPECT_EQ(i.get<bool>("flag"), std::optional<bool>{false});
}

TEST(InfoTest, SetGetInt) {
    info i;
    i.set("n", 42);
    EXPECT_EQ(i.get<int>("n"), std::optional<int>{42});
}

TEST(InfoTest, SetGetNegativeInt) {
    info i;
    i.set("n", -7);
    EXPECT_EQ(i.get<int>("n"), std::optional<int>{-7});
}

TEST(InfoTest, GetTypedAbsentKeyReturnsNullopt) {
    info i;
    EXPECT_EQ(i.get<int>("missing"), std::nullopt);
}

// ── nth_key ───────────────────────────────────────────────────────────────────

TEST(InfoTest, NthKey) {
    info i;
    i.set("alpha", "1");
    i.set("beta", "2");
    EXPECT_EQ(i.nkeys(), 2);
    // MPI_Info keys are ordered by insertion, but the standard doesn't guarantee it —
    // collect both and check by content.
    std::vector<std::string> keys = {i.nth_key(0), i.nth_key(1)};
    EXPECT_TRUE(std::ranges::contains(keys, "alpha"));
    EXPECT_TRUE(std::ranges::contains(keys, "beta"));
}

// ── sentinel / iteration ─────────────────────────────────────────────────────

TEST(InfoTest, SentinelConcept) {
    static_assert(std::sentinel_for<entry_sentinel, entry_iterator>);
}

TEST(InfoTest, EmptyInfoIteratesZeroTimes) {
    info        i;
    int         count = 0;
    for (auto [k, v] : i.entries()) {
        (void)k;
        (void)v;
        ++count;
    }
    EXPECT_EQ(count, 0);
}

TEST(InfoTest, IteratesAllEntries) {
    info i;
    i.set("x", "1");
    i.set("y", "2");
    i.set("z", "3");

    std::vector<std::pair<std::string, std::string>> collected;
    for (auto kv : i.entries()) {
        collected.push_back(kv);
    }
    EXPECT_EQ(collected.size(), 3u);
    // Each key must appear exactly once.
    auto has_key = [&](std::string const& k) {
        return std::ranges::any_of(collected, [&](auto const& p) { return p.first == k; });
    };
    EXPECT_TRUE(has_key("x"));
    EXPECT_TRUE(has_key("y"));
    EXPECT_TRUE(has_key("z"));
}

TEST(InfoTest, EntriesRangeCompatibleWithAlgorithms) {
    info i;
    i.set("a", "1");
    i.set("b", "2");

    // std::ranges::distance works with sentinel ranges.
    EXPECT_EQ(std::ranges::distance(i.entries()), 2);
}

TEST(InfoTest, SetGetThreadLevel) {
    info i;
    i.set("thread_level", ThreadLevel::multiple);
    EXPECT_EQ(i.get<ThreadLevel>("thread_level"), std::optional<ThreadLevel>{ThreadLevel::multiple});
    i.set("thread_level", ThreadLevel::single);
    EXPECT_EQ(i.get<ThreadLevel>("thread_level"), std::optional<ThreadLevel>{ThreadLevel::single});
}

TEST(InfoTest, StructuredBindingsRangeFor) {
    info i;
    i.set("hello", "world");
    for (auto [k, v] : i) { // begin()/end() on i itself
        EXPECT_EQ(k, "hello");
        EXPECT_EQ(v, "world");
    }
}
