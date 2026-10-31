// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

// Uses the installed kamping-v2 package: point-to-point, a variadic collective and a derived datatype.

#include <cstdio>
#include <utility>
#include <vector>

#include <kamping/types/mpi_type_traits.hpp>
#include <kamping/types/struct_type.hpp>
#include <kamping/v2.hpp>

namespace v2 = kamping::v2;

struct Point {
    int    id;
    double x;
};

// Reflected via Boost.PFR, so this also checks that the bundled PFR headers are installed.
template <>
struct kamping::types::mpi_type_traits<Point> : kamping::types::struct_type<Point> {};

int main(int argc, char* argv[]) {
    v2::environment env(argc, argv);
    v2::comm_view   comm{MPI_COMM_WORLD};

    std::vector<int> mine(static_cast<std::size_t>(comm.rank()) + 1, comm.rank());
    auto [_, all] = v2::allgatherv(mine, v2::auto_recv_v<int>(), comm);
    if (static_cast<int>(all.underlying().size()) != comm.size() * (comm.size() + 1) / 2) {
        std::fprintf(stderr, "allgatherv: unexpected size\n");
        return 1;
    }

    v2::type_pool pool;
    if (comm.size() >= 2 && comm.rank() < 2) {
        std::vector<Point> points{{comm.rank(), 1.5}};
        auto [s, r] = v2::sendrecv(
            points | v2::views::with_auto_pool(pool),
            1 - comm.rank(),
            std::vector<Point>{} | v2::views::with_auto_pool(pool) | v2::views::resize,
            comm
        );
        if (r.underlying().size() != 1 || r.underlying()[0].id != 1 - comm.rank()) {
            std::fprintf(stderr, "sendrecv: unexpected data\n");
            return 1;
        }
    }
    return 0;
}
