cmake_minimum_required(VERSION 3.25)
project(dstl LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/kamping-v2-setup.cmake)
if (NOT TARGET kamping::v2)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../kamping-v2 kamping-v2)
endif ()

add_library(dstl INTERFACE)
add_library(dstl::dstl ALIAS dstl)
target_compile_features(dstl INTERFACE cxx_std_20)
target_sources(
    dstl
    INTERFACE FILE_SET
              HEADERS
              BASE_DIRS
              include
              FILES
              include/dstl/algorithm.hpp
              include/dstl/algorithm/all_of.hpp
              include/dstl/algorithm/exclusive_scan.hpp
              include/dstl/algorithm/inclusive_scan.hpp
              include/dstl/algorithm/is_sorted.hpp
              include/dstl/algorithm/max.hpp
              include/dstl/algorithm/min.hpp
              include/dstl/algorithm/reduce.hpp
              include/dstl/algorithm/shift.hpp
              include/dstl/alltoallv.hpp
              include/dstl/default_init_allocator.hpp
              include/dstl/detail/alltoallv_common.hpp
              include/dstl/detail/recv_buffer_utils.hpp
              include/dstl/distribution.hpp
              include/dstl/dstl.hpp
              include/dstl/exchange_layout.hpp
              include/dstl/factoring.hpp
              include/dstl/flat_alltoallv.hpp
              include/dstl/grid_alltoallv.hpp
              include/dstl/grid_comm.hpp
              include/dstl/redistribute.hpp
              include/dstl/request_reply.hpp
              include/dstl/sparse_alltoall.hpp
              include/dstl/tags.hpp
              include/dstl/thread_multiple_comm.hpp
              include/dstl/visualization.hpp
              include/dstl/with_subset.hpp
)
target_link_libraries(dstl INTERFACE kamping::v2)

# Optional OpenMP acceleration for the funneled / thread_multiple re-bin kernels. When not found, those execution
# policies fall back to a serial rebin (correctness is unaffected).
find_package(OpenMP QUIET)
if (OpenMP_CXX_FOUND)
    target_link_libraries(dstl INTERFACE OpenMP::OpenMP_CXX)
endif ()

if (KAMPING_V2_INSTALL)
    install(
        TARGETS dstl
        EXPORT kamping-v2-dstl-targets
        FILE_SET HEADERS
    )
    install(
        EXPORT kamping-v2-dstl-targets
        NAMESPACE dstl::
        DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR}
    )
    configure_file(cmake/kamping-v2-dstl.cmake.in kamping-v2-dstl.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/kamping-v2-dstl.cmake DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR})
endif ()

if (PROJECT_IS_TOP_LEVEL OR KAMPING_V2_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(tests)
    add_subdirectory(examples)
endif ()
