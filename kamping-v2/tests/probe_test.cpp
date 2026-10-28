// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <optional>
#include <ranges>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/p2p/probe.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "mpi/comm.hpp"
#include "mpi/message.hpp"
#include "mpi/p2p/improbe.hpp"
#include "mpi/p2p/iprobe.hpp"
#include "mpi/p2p/mrecv.hpp"
#include "mpi/p2p/recv.hpp"
#include "mpi/p2p/send.hpp"
#include "mpi/status.hpp"

TEST(ProbeTest, iprobe_no_message) {
    bool res = mpi::experimental::iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD);
    EXPECT_FALSE(res);
}

TEST(ProbeTest, improbe_no_message) {
    mpi::experimental::message msg;
    bool                       res = mpi::experimental::improbe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, msg);
    EXPECT_FALSE(res);
}

TEST(ProbeTest, iprobe) {
    static constexpr int         MY_TAG = 1358;
    mpi::experimental::comm_view world{MPI_COMM_WORLD};
    if (world.rank() == 0) {
        mpi::experimental::send(std::views::single(42), 1, MY_TAG);
    } else if (world.rank() == 1) {
        mpi::experimental::status status;
        // first, try probing with the wrong tag a few times
        for (auto i = 0; i < 4; i++) {
            bool probed = mpi::experimental::iprobe(0, 0, MPI_COMM_WORLD);
            EXPECT_FALSE(probed);
        }
        // next, try with the right tag
        while (!mpi::experimental::iprobe(0, MY_TAG, MPI_COMM_WORLD, status)) {
        }
        EXPECT_EQ(status.source(), 0);
        EXPECT_EQ(status.tag(), MY_TAG);
        EXPECT_EQ(status.count<int>(), 1);
        int val = 0;
        mpi::experimental::recv(kamping::v2::views::ref_single(val), 0, MY_TAG, MPI_COMM_WORLD);
        EXPECT_EQ(val, 42);
    }
}

TEST(ProbeTest, improbe) {
    static constexpr int         MY_TAG = 1358;
    mpi::experimental::comm_view world{MPI_COMM_WORLD};
    if (world.rank() == 0) {
        mpi::experimental::send(std::views::single(42), 1, MY_TAG);
    } else if (world.rank() == 1) {
        mpi::experimental::status status;
        // first, try probing with the wrong tag a few times
        mpi::experimental::message msg;
        for (auto i = 0; i < 4; i++) {
            bool probed = mpi::experimental::improbe(0, 0, MPI_COMM_WORLD, msg);
            EXPECT_FALSE(probed);
        }
        // next, try with the right tag
        while (!mpi::experimental::improbe(0, MY_TAG, MPI_COMM_WORLD, msg, status)) {
        }
        EXPECT_EQ(status.source(), 0);
        EXPECT_EQ(status.tag(), MY_TAG);
        EXPECT_EQ(status.count<int>(), 1);
        int val = 0;
        mpi::experimental::mrecv(kamping::v2::views::ref_single(val), msg);
        EXPECT_EQ(val, 42);
    }
}

TEST(ProbeTest, v2_mprobe) {
    static constexpr int         MY_TAG = 1359;
    mpi::experimental::comm_view world{MPI_COMM_WORLD};
    if (world.rank() == 0) {
        std::vector<int> data{1, 2, 3};
        mpi::experimental::send(data, 1, MY_TAG);
    } else if (world.rank() == 1) {
        // defaults: MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD
        auto res = kamping::v2::mprobe();
        EXPECT_EQ(res.source(), 0);
        EXPECT_EQ(res.tag(), MY_TAG);
        EXPECT_EQ(res.count<int>(), 3);
        EXPECT_EQ(res.status().tag(), MY_TAG);
        std::vector<int> buf(3);
        mpi::experimental::mrecv(buf, res);
        EXPECT_EQ(buf, (std::vector<int>{1, 2, 3}));
    }
}

TEST(ProbeTest, v2_improbe) {
    static constexpr int         MY_TAG = 1360;
    mpi::experimental::comm_view world{MPI_COMM_WORLD};
    if (world.rank() == 0) {
        mpi::experimental::send(std::views::single(42), 1, MY_TAG);
    } else if (world.rank() == 1) {
        // first, try probing with the wrong tag a few times
        for (auto i = 0; i < 4; i++) {
            EXPECT_FALSE(kamping::v2::improbe(0, 0, MPI_COMM_WORLD).has_value());
        }
        // next, try with the right tag
        std::optional<kamping::v2::probe_result> res;
        while (!(res = kamping::v2::improbe(0, MY_TAG, MPI_COMM_WORLD))) {
        }
        EXPECT_EQ(res->source(), 0);
        EXPECT_EQ(res->tag(), MY_TAG);
        EXPECT_EQ(res->count<int>(), 1);
        int val = 0;
        mpi::experimental::mrecv(kamping::v2::views::ref_single(val), *res);
        EXPECT_EQ(val, 42);
    }
}
