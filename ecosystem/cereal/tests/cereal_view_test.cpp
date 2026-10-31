// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/buffer.hpp"
#include "kamping/v2/collectives/bcast.hpp"
#include "kamping/v2/ecosystem/cereal.hpp"
#include "kamping/v2/p2p/recv.hpp"
#include "kamping/v2/p2p/send.hpp"
#include "kamping/v2/views/concepts.hpp"

using Map = std::map<std::string, int>;

TEST(CerealView, SendRecv) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2)
        GTEST_SKIP();

    Map const data{{"one", 1}, {"two", 2}, {"forty-two", 42}};

    if (rank == 0) {
        kamping::v2::send(data | kamping::v2::views::serialize, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        auto result = kamping::v2::recv(kamping::v2::views::deserialize<Map>(), 0, 0, MPI_COMM_WORLD);
        EXPECT_EQ(*result, data);
    }
}

// The recv-side serialization_view is a leaf deferred recv buffer (set_recv_count +
// lazy byte-buffer resize). It must participate in the materialize protocol so an
// explicit kamping::v2::materialize() realizes the byte buffer just like mpi_ptr() does.
TEST(CerealView, MaterializeRealizesRecvBuffer) {
    Map const original{{"one", 1}, {"two", 2}, {"forty-two", 42}};

    // Capture the serialized byte stream (what MPI would deliver).
    auto              sview = original | kamping::v2::views::serialize;
    auto const        n     = static_cast<std::size_t>(kamping::v2::count(sview));
    auto const*       src   = static_cast<char const*>(kamping::v2::ptr(std::as_const(sview)));
    std::vector<char> bytes(src, src + n);

    // Recv side: a view over a target object.
    Map  target;
    auto rview = target | kamping::v2::views::serialize;
    static_assert(kamping::v2::deferred_recv_buf<decltype(rview)>);
    static_assert(kamping::v2::has_materialize<decltype(rview)>);

    rview.set_recv_count(static_cast<std::ptrdiff_t>(n));
    kamping::v2::materialize(rview);                       // realizes the byte buffer
    std::memcpy(kamping::v2::ptr(rview), bytes.data(), n); // MPI writes the payload
    rview.unwrap();                                        // deserialize into target

    EXPECT_EQ(target, original);
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
    kamping::v2::bcast(view, 0, MPI_COMM_WORLD);
    view.unwrap(); // deserializes into data on non-root; no-op on root

    EXPECT_EQ(data, (Map{{"one", 1}, {"two", 2}, {"forty-two", 42}}));
}
