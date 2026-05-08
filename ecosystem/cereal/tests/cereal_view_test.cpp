// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <map>
#include <string>

#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/ecosystem/cereal_view.hpp"
#include "kamping/v2/collectives/bcast.hpp"
#include "kamping/v2/p2p/recv.hpp"
#include "kamping/v2/p2p/send.hpp"

using Map = std::map<std::string, int>;

TEST(CerealView, SendRecv) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2)
        GTEST_SKIP();

    Map const data{{"one", 1}, {"two", 2}, {"forty-two", 42}};

    if (rank == 0) {
        kamping::v2::send(data | kamping::v2::views::serialize, 1, 0);
    } else if (rank == 1) {
        auto result = kamping::v2::recv(kamping::v2::views::deserialize<Map>(), 0, 0);
        EXPECT_EQ(*result, data);
    }
}

TEST(CerealView, Bcast) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2)
        GTEST_SKIP();

    Map data;
    if (rank == 0)
        data = {{"one", 1}, {"two", 2}, {"forty-two", 42}};

    auto view = data | kamping::v2::views::serialize;
    kamping::v2::bcast(view, 0);
    view.unwrap(); // deserializes into data on non-root; no-op on root

    EXPECT_EQ(data, (Map{{"one", 1}, {"two", 2}, {"forty-two", 42}}));
}
