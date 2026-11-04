cmake_minimum_required(VERSION 3.25)
project(kamping-v2-cereal LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/kamping-v2-setup.cmake)
if (NOT TARGET kamping::v2)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../kamping-v2 kamping-v2)
endif ()

# If the consumer already made cereal available, FetchContent_MakeAvailable is a no-op.
FetchContent_Declare(
    cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal
    GIT_TAG v1.3.2
    SYSTEM FIND_PACKAGE_ARGS 1.3.2
)
block()
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(JUST_INSTALL_CEREAL ON)
# Installed alongside kamping-v2, which references its export.
set(CEREAL_INSTALL ${KAMPING_V2_INSTALL})
FetchContent_MakeAvailable(cereal)
endblock()

add_library(kamping_v2_cereal INTERFACE)
add_library(kamping::v2::cereal ALIAS kamping_v2_cereal)
target_compile_features(kamping_v2_cereal INTERFACE cxx_std_20)
target_sources(
    kamping_v2_cereal INTERFACE FILE_SET HEADERS BASE_DIRS include FILES include/kamping/v2/ecosystem/cereal.hpp
)
set_target_properties(kamping_v2_cereal PROPERTIES EXPORT_NAME cereal)
target_link_libraries(kamping_v2_cereal INTERFACE kamping::v2 cereal::cereal)

if (KAMPING_V2_INSTALL)
    install(
        TARGETS kamping_v2_cereal
        EXPORT kamping-v2-cereal-targets
        FILE_SET HEADERS
    )
    install(
        EXPORT kamping-v2-cereal-targets
        NAMESPACE kamping::v2::
        DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR}
    )
    install(FILES cmake/kamping-v2-cereal.cmake DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR})
endif ()

if (PROJECT_IS_TOP_LEVEL OR KAMPING_V2_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(examples)
    add_subdirectory(tests)
endif ()
