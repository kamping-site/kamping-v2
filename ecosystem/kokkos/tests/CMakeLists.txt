include(FetchContent)

# KaTestrophe is declared once in the top-level CMakeLists.txt; just make it available here.
FetchContent_MakeAvailable(KaTestrophe)

add_executable(test_kokkos_view kokkos_view_test.cpp)
target_link_libraries(test_kokkos_view PRIVATE kamping_ecosystem_kokkos kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_kokkos_view CORES 2 4)
