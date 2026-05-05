#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <vector>

#include "kamping/ecosystem/kokkos_view.hpp"
#include "kamping/v2/collectives/alltoallv.hpp"
#include "kamping/v2/p2p/recv.hpp"
#include "kamping/v2/p2p/send.hpp"
#include "kamping/v2/views/auto_counts_view.hpp"
#include "kamping/v2/views/auto_displs_view.hpp"
#include "kamping/v2/views/with_counts_view.hpp"


TEST(KokkosView, SendRecv) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2)
        GTEST_SKIP();

    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }

    using matrix_t = Kokkos::View<int**, Kokkos::LayoutRight>;

    if (rank == 0) {
        matrix_t matrix("send_matrix", 4, 5);
        auto host_matrix = Kokkos::create_mirror_view(matrix);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 5; ++j)
                host_matrix(i, j) = 100 + 10 * i + j;
        Kokkos::deep_copy(matrix, host_matrix);

        auto row = Kokkos::subview(matrix, 1, Kokkos::ALL());
        kamping::v2::send(row | kamping::v2::views::kokkos, 1, 0);

    } else if (rank == 1) {
        matrix_t matrix("recv_matrix", 4, 5);
        auto    row = Kokkos::subview(matrix, 2, Kokkos::ALL());
        auto    received = kamping::v2::recv(row | kamping::v2::views::kokkos, 0, 0);
        received.unwrap();
        Kokkos::fence();

        auto host_row = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, row);
        ASSERT_EQ(host_row.extent(0), 5u);
        for (std::size_t j = 0; j < host_row.extent(0); ++j)
            EXPECT_EQ(host_row(j), 110 + static_cast<int>(j));
    }
}

TEST(KokkosView, SendRecvAutoView) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2)
        GTEST_SKIP();

    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }

    if (rank == 0) {
        Kokkos::View<int*, Kokkos::LayoutRight> v("send_auto", 6);
        auto host_v = Kokkos::create_mirror_view(v);
        for (int i = 0; i < 6; ++i)
            host_v(i) = 200 + i;
        Kokkos::deep_copy(v, host_v);
        kamping::v2::send(v | kamping::v2::views::kokkos, 1, 1);

    } else if (rank == 1) {
        auto  received = kamping::v2::recv(kamping::v2::views::auto_kokkos_view<int>(), 0, 1);
        auto& data     = *received.underlying();
        auto  host     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);

        ASSERT_EQ(host.extent(0), 6u);
        for (std::size_t i = 0; i < host.extent(0); ++i)
            EXPECT_EQ(host(i), 200 + static_cast<int>(i));
    }
}

TEST(KokkosView, Alltoallv) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2)
        GTEST_SKIP();

    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }

    using matrix_t = Kokkos::View<int**, Kokkos::LayoutRight>;
    matrix_t send_matrix("send_matrix", static_cast<std::size_t>(size), static_cast<std::size_t>(size));
    Kokkos::deep_copy(send_matrix, rank);

    std::vector<int> send_counts(static_cast<std::size_t>(size));
    for (int dst = 0; dst < size; ++dst)
        send_counts[static_cast<std::size_t>(dst)] = dst + 1;

    matrix_t recv_matrix(
        "recv_matrix",
        static_cast<std::size_t>(size),
        static_cast<std::size_t>(rank + 1)
    );

    auto result = kamping::v2::alltoallv(
        send_matrix | kamping::v2::views::kokkos | kamping::v2::views::with_counts(send_counts)
            | kamping::v2::views::auto_displs(),
        recv_matrix | kamping::v2::views::kokkos | kamping::v2::views::auto_counts()
            | kamping::v2::views::auto_displs()
    );

    auto& rbuf = result.get<1>();
    rbuf.underlying().unwrap();

    auto host_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, recv_matrix);
    ASSERT_EQ(host_recv.extent(0), static_cast<std::size_t>(size));
    ASSERT_EQ(host_recv.extent(1), static_cast<std::size_t>(rank + 1));
    for (int sender = 0; sender < size; ++sender) {
        for (int j = 0; j < rank + 1; ++j) {
            EXPECT_EQ(host_recv(static_cast<std::size_t>(sender), static_cast<std::size_t>(j)), sender);
        }
    }
}
