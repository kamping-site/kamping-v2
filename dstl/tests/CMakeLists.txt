include(FetchContent)
include(GoogleTest)

FetchContent_Declare(
    KaTestrophe
    GIT_REPOSITORY https://github.com/kamping-site/KaTestrophe.cmake
    GIT_TAG v1.0.2
)
FetchContent_MakeAvailable(KaTestrophe)

add_executable(test_dstl_grid_comm grid_comm_test.cpp)
target_link_libraries(test_dstl_grid_comm PRIVATE dstl kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_dstl_grid_comm CORES 1 2 4 8)

add_executable(test_dstl_grid_alltoallv grid_alltoallv_test.cpp)
target_link_libraries(test_dstl_grid_alltoallv PRIVATE dstl kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_dstl_grid_alltoallv CORES 1 2 4 8)

# The flat tests exercise the funneled / thread_multiple policies, which need elevated MPI thread
# support. They therefore use a custom main (thread_multiple_test_main.cpp) that calls
# MPI_Init_thread(MPI_THREAD_MULTIPLE) and link gtest-mpi-listener directly instead of KaTestrophe::main.
add_executable(test_dstl_thread_multiple_comm thread_multiple_comm_test.cpp thread_multiple_test_main.cpp)
target_link_libraries(test_dstl_thread_multiple_comm PRIVATE dstl kamping_v2_warnings gtest-mpi-listener)
katestrophe_add_mpi_test(test_dstl_thread_multiple_comm CORES 1 2 4 8)

add_executable(test_dstl_flat_alltoallv flat_alltoallv_test.cpp thread_multiple_test_main.cpp)
target_link_libraries(test_dstl_flat_alltoallv PRIVATE dstl kamping_v2_warnings gtest-mpi-listener)
katestrophe_add_mpi_test(test_dstl_flat_alltoallv CORES 1 2 4 8)
