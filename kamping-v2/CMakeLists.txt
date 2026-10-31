cmake_minimum_required(VERSION 3.25)
project(kamping-v2-bindings LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/kamping-v2-setup.cmake)
kamping_v2_require_component(mpi-core mpi::core)

kamping_v2_add_header_only_library(kamping_v2 kamping::v2 ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(kamping_v2 PROPERTIES EXPORT_NAME v2)
target_link_libraries(kamping_v2 INTERFACE mpi::core)

kamping_v2_install_component(
    v2
    TARGETS
    kamping_v2
    NAMESPACE
    kamping::
    REQUIRES
    core
)

if (PROJECT_IS_TOP_LEVEL OR KAMPING_V2_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(tests)
    add_subdirectory(examples)
endif ()
