#include <dstl/algorithm.hpp>
#include <gtest/gtest.h>
#include <mpi.h>
#include <kamping/v2/comm.hpp>

TEST(is_sorted, basic) {
  kamping::v2::comm_view comm{MPI_COMM_WORLD};
  
}

TEST(is_sorted, empty) {
  
}
