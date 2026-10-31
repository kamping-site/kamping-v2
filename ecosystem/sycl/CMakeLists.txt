cmake_minimum_required(VERSION 3.25)
project(kamping-v2-sycl LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/kamping-v2-setup.cmake)
kamping_v2_require_component(kamping-v2 kamping::v2)

# SYCL support comes from the compiler (Intel oneAPI DPC++); consumers add it to their targets with
# add_sycl_to_target().
kamping_v2_add_header_only_library(kamping_ecosystem_sycl kamping::ecosystem::sycl ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(kamping_ecosystem_sycl PROPERTIES EXPORT_NAME sycl)
target_link_libraries(kamping_ecosystem_sycl INTERFACE kamping::v2)

kamping_v2_install_component(
    sycl
    TARGETS
    kamping_ecosystem_sycl
    NAMESPACE
    kamping::ecosystem::
    REQUIRES
    v2
)

if (PROJECT_IS_TOP_LEVEL OR KAMPING_V2_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(examples)
endif ()
