// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "kamping/ecosystem/kokkos_view.hpp"
#include "kamping/v2/collectives/alltoallv.hpp"
#include "kamping/v2/environment.hpp"
#include "kamping/v2/views/auto_counts_view.hpp"
#include "kamping/v2/views/auto_displs_view.hpp"
#include "kamping/v2/views/with_counts_view.hpp"
#include "mpi/comm.hpp"

using matrix_t = Kokkos::View<int**, Kokkos::LayoutRight>;

template <typename MatrixT>
void print_matrix(int rank, int size, MatrixT const& matrix, int cols_to_print) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrix);
    std::string out = "rank " + std::to_string(rank) + "\n";
    for (int row = 0; row < size; ++row) {
        out += "  row " + std::to_string(row) + ": [";
        for (int col = 0; col < cols_to_print; ++col) {
            if (col > 0) out += ", ";
            out += std::to_string(h(row, col));
        }
        out += "]\n";
    }
    out += "\n";
    std::cout << out;
}

int main(int argc, char* argv[]) {
    kamping::v2::environment           env(argc, argv);
    mpi::experimental::comm_view const world{MPI_COMM_WORLD};
    Kokkos::initialize(argc, argv);

    {
        int const rank = world.rank();
        int const size = world.size();

        matrix_t send_matrix("send_matrix", static_cast<std::size_t>(size), static_cast<std::size_t>(size));
        Kokkos::deep_copy(send_matrix, rank);

        std::vector<int> send_counts(static_cast<std::size_t>(size));
        for (int dst = 0; dst < size; ++dst) {
            send_counts[static_cast<std::size_t>(dst)] = dst + 1;
        }

        {
            // Example 1: receive into (size , (rank + 1)) matrix
            matrix_t recv_matrix("recv_matrix", static_cast<std::size_t>(size), static_cast<std::size_t>(rank + 1));

            kamping::v2::alltoallv(
                send_matrix | kamping::v2::views::kokkos | kamping::v2::views::with_counts(send_counts)
                    | kamping::v2::views::auto_displs(),
                recv_matrix | kamping::v2::views::kokkos | kamping::v2::views::auto_counts()
                    | kamping::v2::views::auto_displs(),
                world
            );
        }

        {
            // Example 2: receive into a contiguous subview (all columns, first rank+1 rows).
            matrix_t recv_matrix_full(
                "recv_matrix_full",
                static_cast<std::size_t>(size),
                static_cast<std::size_t>(size)
            );
            Kokkos::deep_copy(recv_matrix_full, -1);

            auto recv_subview = Kokkos::subview(
                recv_matrix_full,
                std::pair<std::size_t, std::size_t>{0, static_cast<std::size_t>(rank + 1)},
                Kokkos::ALL()
            );

            kamping::v2::alltoallv(
                send_matrix | kamping::v2::views::kokkos | kamping::v2::views::with_counts(send_counts)
                    | kamping::v2::views::auto_displs(),
                recv_subview | kamping::v2::views::kokkos | kamping::v2::views::auto_counts()
                    | kamping::v2::views::auto_displs(),
                world
            );
        }

        {
            // Example 3: receive into a non-contiguous subview (all rows, first rank+1 columns)
            matrix_t recv_matrix_full(
                "recv_matrix_full",
                static_cast<std::size_t>(size),
                static_cast<std::size_t>(size)
            );
            Kokkos::deep_copy(recv_matrix_full, -1);

            auto recv_subview = Kokkos::subview(
                recv_matrix_full,
                Kokkos::ALL(),
                std::pair<std::size_t, std::size_t>{0, static_cast<std::size_t>(rank + 1)}
            );

            auto result = kamping::v2::alltoallv(
                send_matrix | kamping::v2::views::kokkos | kamping::v2::views::with_counts(send_counts)
                    | kamping::v2::views::auto_displs(),
                recv_subview | kamping::v2::views::kokkos | kamping::v2::views::auto_counts()
                    | kamping::v2::views::auto_displs(),
                world
            );

            auto& sbuf = result.get<0>();
            auto& rbuf = result.get<1>();

            // Get the kokkos_view and unwrap it, for the rbuf this will deep copy back the into the recv_subview
            sbuf.underlying().unwrap();
            rbuf.underlying().unwrap();

            print_matrix(rank, size, send_matrix, size);
            print_matrix(rank, size, recv_matrix_full, size);
        }
    }

    Kokkos::finalize();
    return 0;
}