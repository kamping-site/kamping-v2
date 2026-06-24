// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

/// @file
/// Demonstrates the dstl visualization helpers (`dstl/visualization.hpp`):
///   * `std::format("{}", grid)` — a local, communication-free dump of the grid structure.
///   * `dstl::as_string(comm)`   — a collective per-rank OpenMP-thread table.
///   * `dstl::as_string(grid)`   — a collective grid breakdown annotated with per-rank thread counts.
///
/// Run with, e.g.:  OMP_NUM_THREADS=4 mpirun -np 6 ./example_dstl_grid_visualization
/// The example honors OMP_NUM_THREADS; set it (or place ranks on nodes with different core counts) to
/// see the per-rank thread column change.

#include <format>
#include <iostream>
#include <string>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <mpi.h>

#include "dstl/grid_comm.hpp"
#include "dstl/visualization.hpp"
#include "kamping/v2/environment.hpp"
#include "mpi/comm.hpp"

int main(int argc, char** argv) {
    // dstl::execution_policy::par needs MPI_THREAD_FUNNELED: the main thread issues every MPI call while the OpenMP
    // threads only do node-local work (the rebin / final deposit in dstl::alltoallv).
    kamping::v2::environment          env(argc, argv, mpi::experimental::ThreadLevel::funneled);
    mpi::experimental::comm_view const world{MPI_COMM_WORLD};
    int const                         rank = world.rank();

    // A funneled OpenMP region: no MPI inside, only the local thread team (sized by OMP_NUM_THREADS)
    // reporting itself.
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << std::format("[rank {}] OpenMP active, {} thread(s)\n", rank, omp_get_num_threads());
    }
#else
    std::cout << std::format("[rank {}] built without OpenMP (1 thread)\n", rank);
#endif

    // Build a grid over the world communicator using the default balanced factoring.
    dstl::grid_comm<dstl::execution_policy::par> const grid(world);

    // (1) Local, communication-free structural dump via std::format. Every rank could print this; we
    //     only do so on rank 0 to keep the output tidy (it is identical everywhere).
    if (rank == 0) {
        std::cout << "\n--- std::format(\"{}\", grid)  [local, no communication] ---\n" << std::format("{}", grid);
    }

    // (2) Collective per-rank thread table for the whole communicator. Every rank must call as_string;
    //     the returned string is identical on all ranks, so again only rank 0 prints.
    std::string const thread_table = dstl::as_string(world);
    if (rank == 0) {
        std::cout << "\n--- dstl::as_string(world)  [collective: threads per rank] ---\n" << thread_table;
    }

    // (3) Collective grid breakdown annotated with each rank's thread count, e.g. `0(4)`.
    std::string const grid_dump = dstl::as_string(grid);
    if (rank == 0) {
        std::cout << "\n--- dstl::as_string(grid)  [collective: ranks per level, with threads] ---\n" << grid_dump;
    }

    return 0;
}
