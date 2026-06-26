include(FetchContent)
include(GoogleTest)

FetchContent_Declare(
    KaTestrophe
    GIT_REPOSITORY https://github.com/kamping-site/KaTestrophe.cmake
    GIT_TAG v1.1.0
)
FetchContent_MakeAvailable(KaTestrophe)

katestrophe_add_mpi_main(thread-multiple-mpi-main THREAD_LEVEL MPI_THREAD_MULTIPLE)
add_executable(test_dstl_grid_comm grid_comm_test.cpp)
target_link_libraries(test_dstl_grid_comm PRIVATE dstl kamping_v2_warnings KaTestrophe::main_thread_multiple)
katestrophe_add_mpi_test(test_dstl_grid_comm CORES 1 2 4 8)

add_executable(test_dstl_grid_alltoallv grid_alltoallv_test.cpp)
target_link_libraries(test_dstl_grid_alltoallv PRIVATE dstl kamping_v2_warnings KaTestrophe::main_thread_multiple)
katestrophe_add_mpi_test(test_dstl_grid_alltoallv CORES 1 2 4 8)

# The flat tests exercise the funneled / thread_multiple policies, which need elevated MPI thread
# support, so they link the thread-multiple-mpi-main built above (MPI_Init_thread(MPI_THREAD_MULTIPLE))
# instead of the default KaTestrophe::main.
add_executable(test_dstl_thread_multiple_comm thread_multiple_comm_test.cpp)
target_link_libraries(test_dstl_thread_multiple_comm PRIVATE dstl kamping_v2_warnings KaTestrophe::main_thread_multiple)
katestrophe_add_mpi_test(test_dstl_thread_multiple_comm CORES 1 2 4 8)

add_executable(test_dstl_flat_alltoallv flat_alltoallv_test.cpp)
target_link_libraries(test_dstl_flat_alltoallv PRIVATE dstl kamping_v2_warnings KaTestrophe::main_thread_multiple)
katestrophe_add_mpi_test(test_dstl_flat_alltoallv CORES 1 2 4 8)
