#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/request.hpp"
#include "mpi/p2p/isend.hpp"

TEST(RequestViewTest, ConceptSatisfied) {
    static_assert(mpi::experimental::convertible_to_mpi_handle_ptr<mpi::experimental::request_view, MPI_Request>);
}

TEST(RequestViewTest, IsendWithRequestView) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) GTEST_SKIP();

    std::vector<int> data = {rank};
    MPI_Request      req  = MPI_REQUEST_NULL;
    mpi::experimental::request_view rv{req};

    if (rank == 0) {
        mpi::experimental::isend(data, 1, 0, MPI_COMM_WORLD, rv);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        std::vector<int> recv(1);
        MPI_Recv(recv.data(), 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        EXPECT_EQ(recv[0], 0);
    }
}

TEST(RequestViewTest, WaitViaRequestView) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) GTEST_SKIP();

    std::vector<int> data = {rank};
    MPI_Request      req  = MPI_REQUEST_NULL;

    if (rank == 0) {
        mpi::experimental::isend(data, 1, 0, MPI_COMM_WORLD, req);
        mpi::experimental::request_view{req}.wait();
    } else if (rank == 1) {
        std::vector<int> recv(1);
        MPI_Recv(recv.data(), 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        EXPECT_EQ(recv[0], 0);
    }
}
