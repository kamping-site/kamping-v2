#include <cstddef>
#include <print>
#include <vector>

#include <mpi.h>

#include "kamping/v2/environment.hpp"
#include "kamping/v2/p2p/irecv.hpp"
#include "kamping/v2/p2p/isend.hpp"
#include "kamping/v2/p2p/isendrecv.hpp"
#include "kamping/v2/p2p/recv.hpp"
#include "kamping/v2/p2p/send.hpp"
#include "kamping/v2/p2p/sendrecv.hpp"
#include "kamping/v2/views/resize_view.hpp"
#include "mpi/comm.hpp"

struct my_struct {
    int val;
};

template <>
struct mpi::experimental::buffer_traits<my_struct> {
    static std::ptrdiff_t count(my_struct const&) {
        return 1;
    }
    static int const* ptr(my_struct const& t) {
        return &t.val;
    }
    static int* ptr(my_struct& t) {
        return &t.val;
    }
    static MPI_Datatype type(my_struct const&) {
        return MPI_INT;
    }
};

int main(int argc, char* argv[]) {
    kamping::v2::environment           env(argc, argv);
    mpi::experimental::comm_view const world{MPI_COMM_WORLD};
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    mpi::experimental::send(my_struct{}, MPI_PROC_NULL, 0, MPI_COMM_WORLD);
    if (world.rank() == 0) {
        std::vector<int> v{1, 2, 3, 4};
        kamping::v2::send(std::move(v), 1, world);
    } else if (world.rank() == 1) {
        std::vector<int> v;
        kamping::v2::recv(v | kamping::v2::views::resize);
        std::println("result = {}", v);
    }

    if (world.rank() == 0) {
        std::vector<int> const v{11, 12, 13, 14};
        kamping::v2::isend(v, 1).wait();
    } else if (world.rank() == 1) {
        MPI_Status status;
        auto       v = kamping::v2::irecv(std::vector<int>{10} | kamping::v2::views::resize).wait(&status);
        std::println("v = {}", v);
    }

    // sendrecv: each rank simultaneously sends to the other and receives from the other
    if (world.size() >= 2 && (world.rank() == 0 || world.rank() == 1)) {
        int const        peer      = 1 - static_cast<int>(world.rank());
        std::vector<int> send_data = (world.rank() == 0) ? std::vector<int>{1, 2, 3} : std::vector<int>{4, 5, 6};
        std::vector<int> recv_data;
        auto&& [_, recvd] =
            kamping::v2::sendrecv(send_data, peer, recv_data | kamping::v2::views::resize, peer, world);
        std::println("rank {} recvd = {}", world.rank(), recvd);
    }

    // isendrecv: non-blocking sendrecv, wait() to retrieve the result
    if (world.size() >= 2 && (world.rank() == 0 || world.rank() == 1)) {
        int const        peer      = 1 - static_cast<int>(world.rank());
        std::vector<int> send_data = (world.rank() == 0) ? std::vector<int>{7, 8, 9} : std::vector<int>{10, 11, 12};
        std::vector<int> recv_data;
        auto&& [_2, recvd2] =
            kamping::v2::isendrecv(send_data, peer, recv_data | kamping::v2::views::resize, peer, world).wait();
        std::println("rank {} isendrecv recvd = {}", world.rank(), recvd2);
    }
    return 0;
}
