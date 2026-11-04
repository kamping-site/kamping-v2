include(GoogleTest)

FetchContent_MakeAvailable(KaTestrophe)

add_executable(test_core_arrays arrays_test.cpp)
target_link_libraries(test_core_arrays PRIVATE mpi::core kamping_v2_warnings GTest::gtest_main)
gtest_discover_tests(test_core_arrays)

add_executable(test_core_handle_range handle_range_test.cpp)
target_link_libraries(test_core_handle_range PRIVATE mpi::core kamping_v2_warnings GTest::gtest_main)
gtest_discover_tests(test_core_handle_range)

add_executable(test_core_cache cache_test.cpp)
target_link_libraries(test_core_cache PRIVATE mpi::core kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_core_cache CORES 1 2 4)

add_executable(test_core_group group_test.cpp)
target_link_libraries(test_core_group PRIVATE mpi::core kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_core_group CORES 1 2 4)

add_executable(test_core_info info_test.cpp)
target_link_libraries(test_core_info PRIVATE mpi::core kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_core_info CORES 1)

add_executable(test_core_win win_test.cpp)
target_link_libraries(test_core_win PRIVATE mpi::core kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_core_win CORES 1 2 4)

add_executable(test_core_completion completion_test.cpp)
target_link_libraries(test_core_completion PRIVATE mpi::core kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_core_completion CORES 1 2 4)

add_executable(test_core_handle_vector handle_vector_test.cpp)
target_link_libraries(test_core_handle_vector PRIVATE mpi::core kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_core_handle_vector CORES 1 2)

add_executable(test_core_request_view request_view_test.cpp)
target_link_libraries(test_core_request_view PRIVATE mpi::core kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_core_request_view CORES 1 2)
