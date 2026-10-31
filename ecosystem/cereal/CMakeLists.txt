cmake_minimum_required(VERSION 3.25)
project(kamping-v2-cereal LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/kamping-v2-setup.cmake)
kamping_v2_require_component(kamping-v2 kamping::v2)

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

kamping_v2_add_header_only_library(
    kamping_ecosystem_cereal kamping::ecosystem::cereal ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(kamping_ecosystem_cereal PROPERTIES EXPORT_NAME cereal)
target_link_libraries(kamping_ecosystem_cereal INTERFACE kamping::v2 cereal::cereal)

kamping_v2_install_component(
    cereal
    TARGETS
    kamping_ecosystem_cereal
    NAMESPACE
    kamping::ecosystem::
    REQUIRES
    v2
    DEPENDENCIES
    "find_dependency(cereal 1.3.2)"
)

if (PROJECT_IS_TOP_LEVEL OR KAMPING_V2_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(examples)
    add_subdirectory(tests)
endif ()
