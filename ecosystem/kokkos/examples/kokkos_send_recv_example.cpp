#include <iostream>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "kamping/v2/environment.hpp"
#include "kamping/ecosystem/kokkos_view.hpp"
#include "kamping/v2/p2p/recv.hpp"
#include "kamping/v2/p2p/irecv.hpp"
#include "kamping/v2/p2p/send.hpp"
#include "mpi/comm.hpp"

template <typename View>
void print_row(View const& to_print) {
    using non_const_t = std::remove_const_t<typename View::value_type>;
    // Non-contiguous device views cannot be copied directly to host; pack into
    // a contiguous device buffer first (same pattern as kokkos_view::pack).
    Kokkos::View<non_const_t*, typename View::memory_space> buf(
        Kokkos::view_alloc(typename View::execution_space{}, Kokkos::WithoutInitializing, "print_row_tmp"),
        to_print.extent(0)
    );
    Kokkos::deep_copy(buf, to_print);
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, buf);
    std::cout << '[';
    for (std::size_t j = 0; j < h.extent(0); ++j)
        std::cout << h(j) << ((j + 1 < h.extent(0)) ? ", " : "");
    std::cout << "]\n";
}


int main(int argc, char* argv[]) {
    kamping::v2::environment           env(argc, argv);
    mpi::experimental::comm_view const world{MPI_COMM_WORLD};
    Kokkos::initialize(argc, argv);

    KAMPING_ASSERT(world.size() == 2uz, "This example must be run with exactly 2 ranks.");
    using matrix_t = Kokkos::View<int**, Kokkos::LayoutLeft>;

    // Send and recv subview on 4x5 matrix
    if (world.rank() == 0) {
        matrix_t matrix("send_matrix", 4, 5);
        Kokkos::parallel_for(
            "init_matrix",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {4, 5}),
            KOKKOS_LAMBDA(int i, int j) { matrix(i, j) = 100 + 10 * i + j; }
        );
        // Get a non-contiguous row view.
        auto row = Kokkos::subview(matrix, 1, Kokkos::ALL());
        kamping::v2::send(row | kamping::v2::views::kokkos, 1, 0, world);

    } else if (world.rank() == 1) {
        matrix_t matrix("recv_matrix", 4, 5);
        auto    row      = Kokkos::subview(matrix, 2, Kokkos::ALL());
        auto    req = kamping::v2::irecv(row | kamping::v2::views::kokkos, 0, 0, world);
      
	// Trigger deep copy back to the row view using unwrap()
	req.wait().unwrap();

        print_row(row);

    }

    // Send vector and recv using kamping::views::unpack<int>
    if (world.rank() == 0) {
        Kokkos::View<int*, Kokkos::LayoutRight> v("send_unpack", 6);
        Kokkos::parallel_for(
            "init_v", 6, KOKKOS_LAMBDA(int i) { v(i) = 200 + i; }
        );
        kamping::v2::send(v | kamping::v2::views::kokkos, 1, 0, world);
    } else if (world.rank() == 1) {
        auto  received = kamping::v2::recv(kamping::v2::views::auto_kokkos_view<int>(), 0, 0, world);

        // Get the Kokkos::View, this won't trigger a deep copy, because auto_kokkos_view is contiguous
        auto& data     = *received.underlying();
        print_row(data);
    }

    // Send 4x5 matrix and recv using kamping::v2::views::auto_kokkos_view<int>
    if (world.rank() == 0) {
        matrix_t matrix("send_matrix", 4, 5);
        Kokkos::parallel_for(
            "init_matrix",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {4, 5}),
            KOKKOS_LAMBDA(int i, int j) { matrix(i, j) = 100 + 10 * i + j; }
        );
        kamping::v2::send(matrix | kamping::v2::views::kokkos, 1, 0, world);

    } else if (world.rank() == 1) {
        auto  received = kamping::v2::recv(kamping::v2::views::auto_kokkos_view<int>(), 0, 0, world);

        // Get the Kokkos::View, this won't trigger a deep copy, because auto_kokkos_view is contiguous
        auto& data     = *received.underlying();
        print_row(data);
    }

    Kokkos::finalize();
    return 0;
}
