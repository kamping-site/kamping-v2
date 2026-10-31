include(FetchContent)
include(GoogleTest)

FetchContent_Declare(
    KaTestrophe
    GIT_REPOSITORY https://github.com/kamping-site/KaTestrophe.cmake
    GIT_TAG v1.1.0
)
FetchContent_MakeAvailable(KaTestrophe)

# mpi_core_add_unit_test(<name>): a GoogleTest test without MPI, built from <name>_test.cpp.
function (mpi_core_add_unit_test name)
    add_executable(test_core_${name} ${name}_test.cpp)
    target_link_libraries(test_core_${name} PRIVATE mpi_core kamping_v2_warnings GTest::gtest_main)
    gtest_discover_tests(test_core_${name})
endfunction ()

# mpi_core_add_mpi_test(<name> CORES <n>...): an MPI test built from <name>_test.cpp, run on each core count.
function (mpi_core_add_mpi_test name)
    cmake_parse_arguments(ARG "" "" "CORES" ${ARGN})
    add_executable(test_core_${name} ${name}_test.cpp)
    target_link_libraries(test_core_${name} PRIVATE mpi_core kamping_v2_warnings KaTestrophe::main)
    katestrophe_add_mpi_test(test_core_${name} CORES ${ARG_CORES})
endfunction ()

mpi_core_add_unit_test(arrays)
mpi_core_add_unit_test(handle_range)

mpi_core_add_mpi_test(cache CORES 1 2 4)
mpi_core_add_mpi_test(group CORES 1 2 4)
mpi_core_add_mpi_test(info CORES 1)
mpi_core_add_mpi_test(win CORES 1 2 4)
