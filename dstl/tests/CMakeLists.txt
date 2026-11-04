include(GoogleTest)

FetchContent_MakeAvailable(KaTestrophe)

add_executable(test_dstl_grid_comm grid_comm_test.cpp)
target_link_libraries(test_dstl_grid_comm PRIVATE dstl::dstl kamping_v2_warnings KaTestrophe::main_thread_multiple)
katestrophe_add_mpi_test(test_dstl_grid_comm CORES 1 2 4 8)

add_executable(test_dstl_grid_alltoallv grid_alltoallv_test.cpp)
target_link_libraries(test_dstl_grid_alltoallv PRIVATE dstl::dstl kamping_v2_warnings KaTestrophe::main_thread_multiple)
katestrophe_add_mpi_test(test_dstl_grid_alltoallv CORES 1 2 4 8)

# The flat tests exercise the funneled / thread_multiple policies, which need MPI_THREAD_MULTIPLE, so they link
# KaTestrophe::main_thread_multiple instead of KaTestrophe::main.
add_executable(test_dstl_thread_multiple_comm thread_multiple_comm_test.cpp)
target_link_libraries(
    test_dstl_thread_multiple_comm PRIVATE dstl::dstl kamping_v2_warnings KaTestrophe::main_thread_multiple
)
katestrophe_add_mpi_test(test_dstl_thread_multiple_comm CORES 1 2 4 8)

add_executable(test_dstl_flat_alltoallv flat_alltoallv_test.cpp)
target_link_libraries(test_dstl_flat_alltoallv PRIVATE dstl::dstl kamping_v2_warnings KaTestrophe::main_thread_multiple)
katestrophe_add_mpi_test(test_dstl_flat_alltoallv CORES 1 2 4 8)

add_executable(test_dstl_is_sorted is_sorted_test.cpp)
target_link_libraries(test_dstl_is_sorted PRIVATE dstl::dstl kamping_v2_warnings KaTestrophe::main_thread_single)
katestrophe_add_mpi_test(test_dstl_is_sorted CORES 1 2 4)

add_executable(test_dstl_shift shift_test.cpp)
target_link_libraries(test_dstl_shift PRIVATE dstl::dstl kamping_v2_warnings KaTestrophe::main_thread_single)
katestrophe_add_mpi_test(test_dstl_shift CORES 1 2 4)

add_executable(test_dstl_redistribute redistribute_test.cpp)
target_link_libraries(test_dstl_redistribute PRIVATE dstl::dstl kamping_v2_warnings KaTestrophe::main_thread_single)
katestrophe_add_mpi_test(test_dstl_redistribute CORES 1 2 4)

add_executable(test_dstl_scan scan_test.cpp)
target_link_libraries(test_dstl_scan PRIVATE dstl::dstl kamping_v2_warnings KaTestrophe::main_thread_single)
katestrophe_add_mpi_test(test_dstl_scan CORES 1 2 4)

add_executable(test_dstl_distribution distribution_test.cpp)
target_link_libraries(test_dstl_distribution PRIVATE dstl::dstl kamping_v2_warnings KaTestrophe::main_thread_single)
katestrophe_add_mpi_test(test_dstl_distribution CORES 1 2 4)

add_executable(test_dstl_max max_test.cpp)
target_link_libraries(test_dstl_max PRIVATE dstl::dstl kamping_v2_warnings KaTestrophe::main_thread_single)
katestrophe_add_mpi_test(test_dstl_max CORES 1 2 4 8)

add_executable(test_dstl_min min_test.cpp)
target_link_libraries(test_dstl_min PRIVATE dstl::dstl kamping_v2_warnings KaTestrophe::main_thread_single)
katestrophe_add_mpi_test(test_dstl_min CORES 1 2 4 8)

add_executable(test_dstl_sparse_alltoall sparse_alltoall_test.cpp)
target_link_libraries(test_dstl_sparse_alltoall PRIVATE dstl::dstl kamping_v2_warnings KaTestrophe::main_thread_single)
katestrophe_add_mpi_test(test_dstl_sparse_alltoall CORES 1 2 4 8)

add_executable(test_dstl_request_reply request_reply_test.cpp)
target_link_libraries(test_dstl_request_reply PRIVATE dstl::dstl kamping_v2_warnings KaTestrophe::main_thread_funneled)
katestrophe_add_mpi_test(test_dstl_request_reply CORES 1 2 4 8)
