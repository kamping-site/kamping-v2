cmake_minimum_required(VERSION 3.25)
project(kamping-v2-sycl LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/kamping-v2-setup.cmake)
if (NOT TARGET kamping::v2)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../kamping-v2 kamping-v2)
endif ()

# SYCL support comes from the compiler (Intel oneAPI DPC++); consumers add it to their targets with
# add_sycl_to_target().
add_library(kamping_v2_sycl INTERFACE)
add_library(kamping::v2::sycl ALIAS kamping_v2_sycl)
target_compile_features(kamping_v2_sycl INTERFACE cxx_std_20)
target_sources(kamping_v2_sycl INTERFACE FILE_SET HEADERS BASE_DIRS include FILES include/kamping/v2/ecosystem/sycl.hpp)
set_target_properties(kamping_v2_sycl PROPERTIES EXPORT_NAME sycl)
target_link_libraries(kamping_v2_sycl INTERFACE kamping::v2)

if (KAMPING_V2_INSTALL)
    install(
        TARGETS kamping_v2_sycl
        EXPORT kamping-v2-sycl-targets
        FILE_SET HEADERS
    )
    install(
        EXPORT kamping-v2-sycl-targets
        NAMESPACE kamping::v2::
        DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR}
    )
    install(FILES cmake/kamping-v2-sycl.cmake DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR})
endif ()

if (PROJECT_IS_TOP_LEVEL OR KAMPING_V2_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(examples)
endif ()
