FetchContent_MakeAvailable(KaTestrophe)

add_executable(test_kokkos_view kokkos_view_test.cpp)
target_link_libraries(test_kokkos_view PRIVATE kamping::v2::kokkos kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_kokkos_view CORES 2 4)
