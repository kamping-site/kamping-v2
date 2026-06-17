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
