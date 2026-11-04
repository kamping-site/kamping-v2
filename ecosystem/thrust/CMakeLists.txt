cmake_minimum_required(VERSION 3.25)
project(kamping-v2-thrust LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/kamping-v2-setup.cmake)
if (NOT TARGET kamping::v2)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../kamping-v2 kamping-v2)
endif ()

# Thrust ships with the CUDA Toolkit.
find_package(CUDAToolkit REQUIRED)

add_library(kamping_v2_thrust INTERFACE)
add_library(kamping::v2::thrust ALIAS kamping_v2_thrust)
target_compile_features(kamping_v2_thrust INTERFACE cxx_std_20)
target_sources(
    kamping_v2_thrust
    INTERFACE FILE_SET
              HEADERS
              BASE_DIRS
              include
              FILES
              include/kamping/v2/ecosystem/thrust.hpp
              include/kamping/v2/ecosystem/thrust_buffer_traits.hpp
)
set_target_properties(kamping_v2_thrust PROPERTIES EXPORT_NAME thrust)
target_link_libraries(kamping_v2_thrust INTERFACE kamping::v2 CUDA::cudart)

if (KAMPING_V2_INSTALL)
    install(
        TARGETS kamping_v2_thrust
        EXPORT kamping-v2-thrust-targets
        FILE_SET HEADERS
    )
    install(
        EXPORT kamping-v2-thrust-targets
        NAMESPACE kamping::v2::
        DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR}
    )
    install(FILES cmake/kamping-v2-thrust.cmake DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR})
endif ()

if (PROJECT_IS_TOP_LEVEL OR KAMPING_V2_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(examples)
endif ()
