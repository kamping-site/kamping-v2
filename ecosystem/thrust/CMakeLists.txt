cmake_minimum_required(VERSION 3.25)
project(kamping-v2-thrust LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/kamping-v2-setup.cmake)
kamping_v2_require_component(kamping-v2 kamping::v2)

# Thrust ships with the CUDA Toolkit.
find_package(CUDAToolkit REQUIRED)

kamping_v2_add_header_only_library(
    kamping_ecosystem_thrust kamping::ecosystem::thrust ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(kamping_ecosystem_thrust PROPERTIES EXPORT_NAME thrust)
target_link_libraries(kamping_ecosystem_thrust INTERFACE kamping::v2 CUDA::cudart)

kamping_v2_install_component(
    thrust
    TARGETS
    kamping_ecosystem_thrust
    NAMESPACE
    kamping::ecosystem::
    REQUIRES
    v2
    DEPENDENCIES
    "find_dependency(CUDAToolkit)"
)

if (PROJECT_IS_TOP_LEVEL OR KAMPING_V2_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(examples)
endif ()
