cmake_minimum_required(VERSION 3.25)
project(kamping-ecosystem-cereal LANGUAGES CXX)

include(FetchContent)

# Bootstrap kamping::v2 when not already provided (monorepo sibling path).
# When consumed by a downstream project that already provides kamping::v2, the guard skips this.
# Outside the monorepo without a pre-existing kamping::v2 target, point cmake at the kamping-v2
# directory first, or use FetchContent_Declare(kamping_v2 ...) before including this component.
if (NOT TARGET kamping::v2)
    set(_kamping_v2_local "${CMAKE_CURRENT_SOURCE_DIR}/../../kamping-v2")
    if (NOT EXISTS "${_kamping_v2_local}/CMakeLists.txt")
        message(
            FATAL_ERROR
            "kamping-ecosystem-cereal: kamping::v2 target not found and the sibling monorepo "
            "directory '${_kamping_v2_local}' does not exist. "
            "Provide kamping::v2 before including this component, or build from the monorepo root."
        )
    endif ()
    FetchContent_Declare(kamping_v2 SOURCE_DIR "${_kamping_v2_local}")
    unset(_kamping_v2_local)
    FetchContent_MakeAvailable(kamping_v2)
endif ()

# Declare cereal here so this component is self-contained. If kamping-v1 (or the consumer)
# already made cereal available, FetchContent_MakeAvailable is a no-op.
FetchContent_Declare(
    cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal
    GIT_TAG v1.3.2
    SYSTEM FIND_PACKAGE_ARGS 1.3.2
)
set(JUST_INSTALL_CEREAL ON)
FetchContent_MakeAvailable(cereal)

add_library(kamping_ecosystem_cereal INTERFACE)
target_include_directories(kamping_ecosystem_cereal INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kamping_ecosystem_cereal INTERFACE kamping::v2 cereal::cereal)
add_library(kamping::ecosystem::cereal ALIAS kamping_ecosystem_cereal)

if (PROJECT_IS_TOP_LEVEL OR KAMPING_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(examples)
endif ()
