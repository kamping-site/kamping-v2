// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/comm.hpp"
#include "kamping/v2/p2p/mrecv.hpp"
#include "kamping/v2/p2p/probe.hpp"
#include "kamping/v2/p2p/send.hpp"
#include "kamping/v2/request.hpp"
#include "kamping/v2/status.hpp"
#include "kamping/v2/views.hpp"

namespace {
std::vector<int> const expected_data{1, 2, 3, 4, 5};

/// Rank 0 sends `expected_data` to rank 1 with `tag`; returns true on rank 1, which should receive it.
bool send_to_rank_one(int tag) {
    kamping::v2::comm_view world{MPI_COMM_WORLD};
    if (world.rank() == 0) {
        kamping::v2::send(expected_data, 1, tag);
    }
    return world.rank() == 1;
}
} // namespace

TEST(V2MrecvTest, mrecv_resizes_deferred_buffer) {
    if (send_to_rank_one(1)) {
        auto             probe = kamping::v2::mprobe(0, 1, MPI_COMM_WORLD);
        std::vector<int> buf;
        kamping::v2::mrecv(buf | kamping::v2::views::resize, std::move(probe));
        EXPECT_EQ(buf, expected_data);
    }
}

TEST(V2MrecvTest, mrecv_into_presized_buffer) {
    if (send_to_rank_one(2)) {
        auto                probe = kamping::v2::mprobe(0, 2, MPI_COMM_WORLD);
        std::vector<int>    buf(expected_data.size());
        kamping::v2::status status;
        kamping::v2::mrecv(buf, std::move(probe), status);
        EXPECT_EQ(buf, expected_data);
        EXPECT_EQ(status.source(), 0);
        EXPECT_EQ(status.tag(), 2);
        EXPECT_EQ(status.count<int>(), static_cast<int>(expected_data.size()));
    }
}

TEST(V2MrecvTest, mrecv_returns_owned_buffer) {
    if (send_to_rank_one(3)) {
        auto probe = kamping::v2::mprobe(0, 3, MPI_COMM_WORLD);
        auto buf   = kamping::v2::mrecv(std::vector<int>(expected_data.size()), std::move(probe));
        EXPECT_EQ(buf, expected_data);
    }
}

TEST(V2MrecvTest, imrecv_resizes_deferred_buffer) {
    if (send_to_rank_one(4)) {
        std::optional<kamping::v2::probe_result> probe;
        while (!(probe = kamping::v2::improbe(0, 4, MPI_COMM_WORLD))) {
        }
        std::vector<int> buf;
        auto             res = kamping::v2::imrecv(buf | kamping::v2::views::resize, std::move(*probe));
        res.wait();
        EXPECT_EQ(buf, expected_data);
    }
}

TEST(V2MrecvTest, imrecv_with_external_request) {
    if (send_to_rank_one(5)) {
        auto                 probe = kamping::v2::mprobe(0, 5, MPI_COMM_WORLD);
        std::vector<int>     buf;
        kamping::v2::request req;
        kamping::v2::imrecv(req, buf | kamping::v2::views::resize, std::move(probe));
        req.wait();
        EXPECT_EQ(buf, expected_data);
    }
}
