// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

// The code between the [docs:...] markers is shown in docs/guide/getting-started.md.

#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

#include <mpi.h>

// [docs:include]
#include <kamping/v2.hpp>

namespace v2    = kamping::v2;
namespace views = kamping::v2::views;
// [docs:include]

int main(int argc, char* argv[]) {
    // [docs:setup]
    v2::environment     env(argc, argv);
    v2::comm_view const world{MPI_COMM_WORLD};
    // [docs:setup]

    // [docs:send-recv]
    if (world.rank() == 0) {
        std::vector<int> data(10);
        std::iota(data.begin(), data.end(), 0);
        // pointer, count and MPI_INT are taken from the vector
        v2::send(data, 1, world);
    } else if (world.rank() == 1) {
        // the size of the message is not known here, so probe for it and resize before receiving
        std::vector<int> data;
        v2::recv(data | views::resize, world);
        std::cout << "rank 1 received " << data.size() << " elements\n";
    }
    // [docs:send-recv]

    // [docs:allgatherv]
    std::vector<int> mine(static_cast<std::size_t>(world.rank()) + 1, world.rank());
    auto [_, all] = v2::allgatherv(mine, v2::auto_recv_v<int>(), world);
    // all == {0, 1, 1, 2, 2, 2, ...}: the counts were exchanged,
    // the displacements computed and `all` sized to fit
    // [docs:allgatherv]

    if (world.rank() == 0) {
        std::cout << "allgatherv received " << all.size() << " elements\n";
    }
    return 0;
}
