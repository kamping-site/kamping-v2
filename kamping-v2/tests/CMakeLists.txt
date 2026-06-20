include(FetchContent)
include(GoogleTest)

# KaTestrophe is declared once in the top-level CMakeLists.txt; just make it available here.
FetchContent_MakeAvailable(KaTestrophe)

# Unit tests (no MPI)
add_executable(test_v2_auto_counts_view auto_counts_view_test.cpp)
target_link_libraries(test_v2_auto_counts_view PRIVATE kamping_v2 kamping_v2_warnings GTest::gtest_main GTest::gmock)
gtest_discover_tests(test_v2_auto_counts_view)

add_executable(test_v2_sentinels sentinels_test.cpp)
target_link_libraries(test_v2_sentinels PRIVATE kamping_v2 kamping_v2_warnings GTest::gtest_main GTest::gmock)
gtest_discover_tests(test_v2_sentinels)

add_executable(test_v2_flatten_v_view flatten_v_view_test.cpp)
target_link_libraries(test_v2_flatten_v_view PRIVATE kamping_v2 kamping_v2_warnings GTest::gtest_main GTest::gmock)
gtest_discover_tests(test_v2_flatten_v_view)

add_executable(test_v2_materialize materialize_test.cpp)
target_link_libraries(test_v2_materialize PRIVATE kamping_v2 kamping_v2_warnings GTest::gtest_main GTest::gmock)
gtest_discover_tests(test_v2_materialize)

# MPI tests
add_executable(test_v2_type_pool v2_type_pool_test.cpp)
target_link_libraries(test_v2_type_pool PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_type_pool CORES 1 2)

add_executable(test_v2_request_view request_view_test.cpp)
target_link_libraries(test_v2_request_view PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_request_view CORES 1 2)

add_executable(test_v2_info info_test.cpp)
target_link_libraries(test_v2_info PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_info CORES 1)

add_executable(test_v2_group group_test.cpp)
target_link_libraries(test_v2_group PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_group CORES 1 2 4)

if(MPI_CXX_VERSION VERSION_GREATER_EQUAL "4.0")
    add_executable(test_v2_session session_test.cpp)
    target_link_libraries(test_v2_session PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
    katestrophe_add_mpi_test(test_v2_session CORES 1 2)
endif()

add_executable(test_v2_allgatherv v2_allgatherv_test.cpp)
target_link_libraries(test_v2_allgatherv PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_allgatherv CORES 1 2 4)

add_executable(test_v2_alltoall v2_alltoall_test.cpp)
target_link_libraries(test_v2_alltoall PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_alltoall CORES 1 2 4)

add_executable(test_v2_alltoallv v2_alltoallv_test.cpp)
target_link_libraries(test_v2_alltoallv PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_alltoallv CORES 1 2 4)

add_executable(test_v2_allreduce v2_allreduce_test.cpp)
target_link_libraries(test_v2_allreduce PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_allreduce CORES 1 2 4)

add_executable(test_v2_reduce v2_reduce_test.cpp)
target_link_libraries(test_v2_reduce PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_reduce CORES 1 2 4)

add_executable(test_v2_gather v2_gather_test.cpp)
target_link_libraries(test_v2_gather PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_gather CORES 1 2 4)

add_executable(test_v2_gatherv v2_gatherv_test.cpp)
target_link_libraries(test_v2_gatherv PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_gatherv CORES 1 2 4)

add_executable(test_v2_scatter v2_scatter_test.cpp)
target_link_libraries(test_v2_scatter PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_scatter CORES 1 2 4)

add_executable(test_v2_scatterv v2_scatterv_test.cpp)
target_link_libraries(test_v2_scatterv PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_scatterv CORES 1 2 4)

add_executable(test_v2_scan v2_scan_test.cpp)
target_link_libraries(test_v2_scan PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_scan CORES 1 2 4)

add_executable(test_v2_exscan v2_exscan_test.cpp)
target_link_libraries(test_v2_exscan PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_exscan CORES 1 2 4)

add_executable(test_v2_recv v2_recv_test.cpp)
target_link_libraries(test_v2_recv PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_recv CORES 2 4)
