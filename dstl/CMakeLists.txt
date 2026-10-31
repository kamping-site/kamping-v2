cmake_minimum_required(VERSION 3.25)
project(dstl LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/kamping-v2-setup.cmake)
kamping_v2_require_component(kamping-v2 kamping::v2)

kamping_v2_add_header_only_library(dstl dstl::dstl ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dstl INTERFACE kamping::v2)

# Optional OpenMP acceleration for the funneled / thread_multiple re-bin kernels. When not found, those execution
# policies fall back to a serial rebin (correctness is unaffected).
set(dstl_dependencies)
find_package(OpenMP QUIET)
if (OpenMP_CXX_FOUND)
    target_link_libraries(dstl INTERFACE OpenMP::OpenMP_CXX)
    list(APPEND dstl_dependencies "find_dependency(OpenMP COMPONENTS CXX)")
endif ()

kamping_v2_install_component(
    dstl
    TARGETS dstl
    NAMESPACE dstl::
    REQUIRES v2
    DEPENDENCIES ${dstl_dependencies}
)

if (PROJECT_IS_TOP_LEVEL OR KAMPING_V2_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(tests)
    add_subdirectory(examples)
endif ()
