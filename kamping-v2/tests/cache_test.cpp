// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <concepts>
#include <utility>

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/attribute.hpp"
#include "mpi/cache.hpp"
#include "mpi/comm.hpp"
#include "mpi/win.hpp"

namespace {
/// Neither copyable nor movable, so it can only be built in place; counts its destructions.
struct tracked {
    static inline int destroyed = 0;

    int value;

    explicit tracked(int v) : value(v) {}
    tracked(tracked const&)            = delete;
    tracked& operator=(tracked const&) = delete;
    ~tracked() {
        ++destroyed;
    }
};

using comm_cache = mpi::experimental::cache<mpi::experimental::attribute::comm, tracked>;

MPI_Comm dup_world() {
    MPI_Comm c = MPI_COMM_NULL;
    MPI_Comm_dup(MPI_COMM_WORLD, &c);
    return c;
}
} // namespace

class CacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        tracked::destroyed = 0;
    }
};

TEST_F(CacheTest, CreatesOncePerObject) {
    comm_cache c;
    int        calls = 0;
    auto       make  = [&] {
        ++calls;
        return tracked(42);
    };
    tracked& first  = c.get_or_create(MPI_COMM_WORLD, make);
    tracked& second = c.get_or_create(MPI_COMM_WORLD, make);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(&first, &second);
    EXPECT_EQ(first.value, 42);
    EXPECT_EQ(c.size(), 1u);
    EXPECT_EQ(c.find(MPI_COMM_WORLD), &first);
}

TEST_F(CacheTest, AcceptsHandleWrappers) {
    comm_cache                   c;
    mpi::experimental::comm_view world{MPI_COMM_WORLD};
    tracked&                     value = c.get_or_create(world, [] { return tracked(1); });
    EXPECT_EQ(c.find(MPI_COMM_WORLD), &value);
}

TEST_F(CacheTest, DuplicatedCommunicatorDoesNotInherit) {
    comm_cache c;
    std::ignore   = c.get_or_create(MPI_COMM_WORLD, [] { return tracked(1); });
    MPI_Comm copy = dup_world();
    EXPECT_EQ(c.find(copy), nullptr);
    EXPECT_EQ(c.get_or_create(copy, [] { return tracked(2); }).value, 2);
    EXPECT_EQ(c.size(), 2u);
    MPI_Comm_free(&copy);
}

TEST_F(CacheTest, FreeingObjectDestroysValue) {
    comm_cache c;
    MPI_Comm   copy = dup_world();
    std::ignore     = c.get_or_create(copy, [] { return tracked(1); });
    MPI_Comm_free(&copy);
    EXPECT_EQ(tracked::destroyed, 1);
    EXPECT_TRUE(c.empty());
}

TEST_F(CacheTest, EraseDestroysValue) {
    comm_cache c;
    std::ignore = c.get_or_create(MPI_COMM_WORLD, [] { return tracked(1); });
    EXPECT_TRUE(c.erase(MPI_COMM_WORLD));
    EXPECT_EQ(tracked::destroyed, 1);
    EXPECT_EQ(c.find(MPI_COMM_WORLD), nullptr);
    EXPECT_FALSE(c.erase(MPI_COMM_WORLD));
}

TEST_F(CacheTest, DestroyingCacheDestroysValuesAndDetaches) {
    MPI_Comm copy = dup_world();
    {
        comm_cache c;
        std::ignore = c.get_or_create(MPI_COMM_WORLD, [] { return tracked(1); });
        std::ignore = c.get_or_create(copy, [] { return tracked(2); });
    }
    EXPECT_EQ(tracked::destroyed, 2);
    // The attribute was removed with the cache, so freeing the communicator must not call back into it.
    MPI_Comm_free(&copy);
    EXPECT_EQ(tracked::destroyed, 2);
}

TEST_F(CacheTest, IndependentCachesDoNotInterfere) {
    comm_cache a;
    comm_cache b;
    std::ignore = a.get_or_create(MPI_COMM_WORLD, [] { return tracked(1); });
    EXPECT_EQ(b.find(MPI_COMM_WORLD), nullptr);
    EXPECT_EQ(b.get_or_create(MPI_COMM_WORLD, [] { return tracked(2); }).value, 2);
    EXPECT_EQ(a.find(MPI_COMM_WORLD)->value, 1);
}

TEST_F(CacheTest, MovedCacheKeepsValuesAndCallbacks) {
    MPI_Comm   copy = dup_world();
    comm_cache source;
    std::ignore = source.get_or_create(copy, [] { return tracked(1); });
    comm_cache target(std::move(source));
    EXPECT_EQ(target.find(copy)->value, 1);
    // The delete callback must reach the moved-to cache.
    MPI_Comm_free(&copy);
    EXPECT_EQ(tracked::destroyed, 1);
    EXPECT_TRUE(target.empty());
}

TEST_F(CacheTest, MoveAssignmentDestroysPreviousValues) {
    comm_cache a;
    comm_cache b;
    std::ignore = a.get_or_create(MPI_COMM_WORLD, [] { return tracked(1); });
    std::ignore = b.get_or_create(MPI_COMM_WORLD, [] { return tracked(2); });
    a           = std::move(b);
    EXPECT_EQ(tracked::destroyed, 1);
    EXPECT_EQ(a.find(MPI_COMM_WORLD)->value, 2);
}

TEST_F(CacheTest, DatatypeKind) {
    MPI_Datatype t = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(2, MPI_INT, &t);
    mpi::experimental::cache<mpi::experimental::attribute::type, tracked> c;
    EXPECT_EQ(c.get_or_create(t, [] { return tracked(7); }).value, 7);
    MPI_Type_free(&t);
    EXPECT_EQ(tracked::destroyed, 1);
    EXPECT_TRUE(c.empty());
}

TEST_F(CacheTest, WindowKind) {
    auto [w, local] = mpi::experimental::win_allocate<int>(1, MPI_COMM_WORLD);
    mpi::experimental::cache<mpi::experimental::attribute::win, tracked> c;
    EXPECT_EQ(c.get_or_create(w, [] { return tracked(3); }).value, 3);
    w.free();
    EXPECT_EQ(tracked::destroyed, 1);
    EXPECT_TRUE(c.empty());
}

TEST_F(CacheTest, CachesCollectivelyCreatedWindows) {
    // The intended use: one window per communicator, created on first use and freed with the cache.
    mpi::experimental::cache<mpi::experimental::attribute::comm, mpi::experimental::win> windows;
    MPI_Win                                                                              first = MPI_WIN_NULL;
    for (int i = 0; i < 3; ++i) {
        auto& w = windows.get_or_create(MPI_COMM_WORLD, [] {
            return mpi::experimental::win_allocate<int>(1, MPI_COMM_WORLD).first;
        });
        if (i == 0) {
            first = mpi::experimental::handle(w);
        }
        EXPECT_EQ(mpi::experimental::handle(w), first);
    }
    EXPECT_EQ(windows.size(), 1u);
}

// ── keyval: values owned by the objects' attributes ──────────────────────────

namespace {
using comm_keyval = mpi::experimental::keyval<mpi::experimental::attribute::comm, tracked>;
} // namespace

class KeyvalTest : public ::testing::Test {
protected:
    void SetUp() override {
        tracked::destroyed = 0;
    }
};

TEST_F(KeyvalTest, CreatesOncePerObject) {
    comm_keyval key;
    int         calls = 0;
    auto        make  = [&] {
        ++calls;
        return tracked(42);
    };
    MPI_Comm copy   = dup_world();
    tracked& first  = key.get_or_create(copy, make);
    tracked& second = key.get_or_create(copy, make);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(&first, &second);
    EXPECT_EQ(key.find(copy), &first);
    MPI_Comm_free(&copy);
}

TEST_F(KeyvalTest, FreeingObjectDestroysValue) {
    comm_keyval key;
    MPI_Comm    copy = dup_world();
    std::ignore      = key.get_or_create(copy, [] { return tracked(1); });
    MPI_Comm_free(&copy);
    EXPECT_EQ(tracked::destroyed, 1);
}

TEST_F(KeyvalTest, EraseDestroysValue) {
    comm_keyval key;
    std::ignore = key.get_or_create(MPI_COMM_WORLD, [] { return tracked(1); });
    EXPECT_TRUE(key.erase(MPI_COMM_WORLD));
    EXPECT_EQ(tracked::destroyed, 1);
    EXPECT_EQ(key.find(MPI_COMM_WORLD), nullptr);
    EXPECT_FALSE(key.erase(MPI_COMM_WORLD));
}

TEST_F(KeyvalTest, DestroyingKeyKeepsValuesUntilObjectIsFreed) {
    MPI_Comm copy = dup_world();
    {
        comm_keyval key;
        std::ignore = key.get_or_create(copy, [] { return tracked(1); });
    }
    EXPECT_EQ(tracked::destroyed, 0);
    MPI_Comm_free(&copy);
    EXPECT_EQ(tracked::destroyed, 1);
}

TEST_F(KeyvalTest, IndependentKeysDoNotInterfere) {
    comm_keyval a;
    comm_keyval b;
    MPI_Comm    copy = dup_world();
    std::ignore      = a.get_or_create(copy, [] { return tracked(1); });
    EXPECT_EQ(b.find(copy), nullptr);
    EXPECT_EQ(b.get_or_create(copy, [] { return tracked(2); }).value, 2);
    EXPECT_EQ(a.find(copy)->value, 1);
    MPI_Comm_free(&copy);
    EXPECT_EQ(tracked::destroyed, 2);
}

TEST_F(KeyvalTest, MovedKeyStillFindsValues) {
    comm_keyval source;
    MPI_Comm    copy = dup_world();
    std::ignore      = source.get_or_create(copy, [] { return tracked(1); });
    comm_keyval target(std::move(source));
    EXPECT_EQ(target.find(copy)->value, 1);
    MPI_Comm_free(&copy);
    EXPECT_EQ(tracked::destroyed, 1);
}

TEST_F(KeyvalTest, DatatypeKind) {
    MPI_Datatype t = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(2, MPI_INT, &t);
    mpi::experimental::keyval<mpi::experimental::attribute::type, tracked> key;
    EXPECT_EQ(key.get_or_create(t, [] { return tracked(7); }).value, 7);
    MPI_Type_free(&t);
    EXPECT_EQ(tracked::destroyed, 1);
}

TEST_F(KeyvalTest, StaticKeyvalOutlivesFinalize) {
    // Destroyed at program exit, after MPI_Finalize: the destructor must not free the key then.
    static comm_keyval key;
    MPI_Comm           copy = dup_world();
    std::ignore             = key.get_or_create(copy, [] { return tracked(1); });
    MPI_Comm_free(&copy);
    EXPECT_EQ(tracked::destroyed, 1);
}

TEST(AttributeConstnessTest, CachePropagatesConstKeyvalDoesNot) {
    using comm_keyval_t = mpi::experimental::keyval<mpi::experimental::attribute::comm, tracked>;
    // cache owns its values: a const cache gives read-only access.
    static_assert(std::same_as<decltype(std::declval<comm_cache&>().find(MPI_COMM_WORLD)), tracked*>);
    static_assert(std::same_as<decltype(std::declval<comm_cache const&>().find(MPI_COMM_WORLD)), tracked const*>);
    // keyval is only a key: constness does not reach the values.
    static_assert(std::same_as<decltype(std::declval<comm_keyval_t const&>().find(MPI_COMM_WORLD)), tracked*>);

    comm_cache c;
    std::ignore             = c.get_or_create(MPI_COMM_WORLD, [] { return tracked(5); });
    comm_cache const& c_ref = c;
    EXPECT_EQ(c_ref.find(MPI_COMM_WORLD)->value, 5);
}
