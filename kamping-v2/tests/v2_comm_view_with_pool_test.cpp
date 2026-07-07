// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/types/contiguous_type.hpp"
#include "kamping/v2/collectives/alltoallv.hpp"
#include "kamping/v2/collectives/bcast.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/type_pool.hpp"
#include "kamping/v2/views.hpp"
#include "mpi/comm.hpp"

struct Payload {
    int    id;
    double value;
};

template <>
struct kamping::types::mpi_type_traits<Payload> : public kamping::types::byte_serialized<Payload> {};

// comm_view_with_pool can be constructed from MPI_Comm + pool and from comm_view + pool.
TEST(CommViewWithPoolTest, Construction) {
    kamping::v2::type_pool           pool;
    kamping::v2::comm_view_with_pool from_raw{MPI_COMM_WORLD, pool};
    EXPECT_EQ(from_raw.mpi_handle(), MPI_COMM_WORLD);
    EXPECT_EQ(&from_raw.pool(), &pool);

    mpi::experimental::comm_view     cv{MPI_COMM_WORLD};
    kamping::v2::comm_view_with_pool from_view{cv, pool};
    EXPECT_EQ(from_view.mpi_handle(), MPI_COMM_WORLD);
    EXPECT_EQ(&from_view.pool(), &pool);
}

// comm_view_with_pool is copyable and the copy shares the same pool.
TEST(CommViewWithPoolTest, Copyable) {
    kamping::v2::type_pool           pool;
    kamping::v2::comm_view_with_pool env{MPI_COMM_WORLD, pool};
    kamping::v2::comm_view_with_pool copy = env;
    EXPECT_EQ(&copy.pool(), &pool);
    EXPECT_EQ(copy.mpi_handle(), MPI_COMM_WORLD);
}

// comm_accessors are forwarded: rank() and size() work.
TEST(CommViewWithPoolTest, CommAccessors) {
    kamping::v2::type_pool           pool;
    kamping::v2::comm_view_with_pool env{MPI_COMM_WORLD, pool};
    EXPECT_GE(env.size(), 1);
    EXPECT_GE(env.rank(), 0);
    EXPECT_LT(env.rank(), env.size());
}

// views::with_auto_pool accepts comm_view_with_pool directly.
TEST(CommViewWithPoolTest, WithAutoPoolAcceptsEnv) {
    kamping::v2::type_pool           pool;
    kamping::v2::comm_view_with_pool env{MPI_COMM_WORLD, pool};
    std::vector<Payload>             v{{1, 2.0}};
    auto                             view = v | kamping::v2::views::with_auto_pool(env);
    EXPECT_TRUE(pool.find<Payload>().has_value());
    EXPECT_EQ(mpi::experimental::type(view), *pool.find<Payload>());
}

// views::with_pool (pre-registered) also accepts comm_view_with_pool.
TEST(CommViewWithPoolTest, WithPoolAcceptsEnv) {
    kamping::v2::type_pool           pool;
    MPI_Datatype                     registered = pool.register_type<Payload>();
    kamping::v2::comm_view_with_pool env{MPI_COMM_WORLD, pool};
    std::vector<Payload>             v{{1, 2.0}};
    auto                             view = v | kamping::v2::views::with_pool(env);
    EXPECT_EQ(mpi::experimental::type(view), registered);
}

// Integration: bcast of custom struct via comm_view_with_pool.
TEST(CommViewWithPoolTest, BcastIntegration) {
    kamping::v2::type_pool           pool;
    kamping::v2::comm_view_with_pool env{MPI_COMM_WORLD, pool};
    std::vector<Payload>             v;
    if (env.rank() == 0) {
        v.emplace_back(Payload{7, 3.14});
    } else {
        v.emplace_back(Payload{0, 0.0});
    }
    kamping::v2::bcast(v | kamping::v2::views::with_auto_pool(env), 0, env);
    EXPECT_EQ(v[0].id, 7);
    EXPECT_DOUBLE_EQ(v[0].value, 3.14);
}

// auto_recv<T>(env) and auto_recv_v<T>(env) accept comm_view_with_pool.
TEST(CommViewWithPoolTest, AutoRecvFactories) {
    kamping::v2::type_pool           pool;
    kamping::v2::comm_view_with_pool env{MPI_COMM_WORLD, pool};

    auto recv_buf   = kamping::v2::auto_recv<Payload>(env);
    auto recv_buf_v = kamping::v2::auto_recv_v<Payload>(env);
    // Both factories register the type in the pool as a side effect.
    EXPECT_TRUE(pool.find<Payload>().has_value());
}
