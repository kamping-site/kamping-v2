#include <iostream>
#include <string>
#include <unordered_map>

#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <mpi.h>

#include "kamping/ecosystem/cereal_view.hpp"
#include "kamping/v2/environment.hpp"
#include "kamping/v2/p2p/isend.hpp"
#include "kamping/v2/p2p/recv.hpp"
#include "kamping/v2/p2p/send.hpp"
#include "mpi/comm.hpp"

int main(int argc, char* argv[]) {
    kamping::v2::environment           env(argc, argv);
    mpi::experimental::comm_view const world{MPI_COMM_WORLD};

    // Blocking serialized send/recv of an unordered_map
    if (world.rank() == 0) {
        std::unordered_map<std::string, int> map{{"one", 1}, {"two", 2}, {"forty-two", 42}};
        kamping::v2::send(map | kamping::v2::views::serialize, 1, world);
    } else if (world.rank() == 1) {
        auto result =
            kamping::v2::recv(kamping::v2::views::deserialize<std::unordered_map<std::string, int>>(), world);
        std::cout << "result = {";
        for (auto const& [k, v] : *result) std::cout << k << ": " << v << ", ";
        std::cout << "}\n";
    }

    // Non-blocking serialized send
    if (world.rank() == 0) {
        std::unordered_map<std::string, int> map{{"ett", 1}, {"två", 2}, {"fyrtio-två", 42}};
        kamping::v2::isend(map | kamping::v2::views::serialize, 1, world).wait();
    } else if (world.rank() == 1) {
        auto result = kamping::v2::recv(kamping::v2::views::deserialize<std::unordered_map<std::string, int>>(), world);
        std::cout << "result = {";
        for (auto const& [k, v] : *result) std::cout << k << ": " << v << ", ";
        std::cout << "}\n";
    }

    return 0;
}
