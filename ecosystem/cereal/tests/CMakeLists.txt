FetchContent_MakeAvailable(KaTestrophe)

add_executable(test_cereal_view cereal_view_test.cpp)
target_link_libraries(test_cereal_view PRIVATE kamping::v2::cereal kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_cereal_view CORES 2 4)
