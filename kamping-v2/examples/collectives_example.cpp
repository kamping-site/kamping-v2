#include <iostream>
#include <vector>

#include <mpi.h>

#include "kamping/v2/collectives/allgather.hpp"
#include "kamping/v2/collectives/allgatherv.hpp"
#include "kamping/v2/collectives/bcast.hpp"
#include "kamping/v2/environment.hpp"
#include "kamping/v2/views.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "kamping/v2/views/resize_view.hpp"
#include "mpi/comm.hpp"

int main(int argc, char* argv[]) {
    kamping::v2::environment           env(argc, argv);
    mpi::experimental::comm_view const world{MPI_COMM_WORLD};
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    {
        int val = 0;
        if (world.rank() == 0) {
            val = 42;
        }
        kamping::v2::bcast(kamping::v2::views::ref_single(val), 0, world);
        std::cout << "[R" << world.rank() << "] bcast result=" << val << '\n';
    }
    {
        std::vector<int> sbuf{static_cast<int>(world.rank()), static_cast<int>(world.rank())};
        auto             v = kamping::v2::allgather(sbuf, std::vector<int>{} | kamping::v2::views::resize, world).recv;
        std::cout << "allgather v=[";
        for (std::size_t i = 0; i < v.size(); ++i) { if (i) std::cout << ", "; std::cout << v[i]; }
        std::cout << "]\n";
    }
    {
        std::vector<int> sbuf{static_cast<int>(world.rank()), static_cast<int>(world.rank())};
        auto             v =
            kamping::v2::allgatherv(
                sbuf,
                std::vector<int>{} | kamping::v2::views::auto_counts() | kamping::v2::views::auto_displs()
                    | kamping::v2::views::resize_v,
                world
            )
                .recv;
        std::cout << "allgatherv v=[";
        for (std::size_t i = 0; i < v.size(); ++i) { if (i) std::cout << ", "; std::cout << v[i]; }
        std::cout << "]\n";
    }
    return 0;
}
