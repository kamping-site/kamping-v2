cmake_minimum_required(VERSION 3.25)
project(kamping-ecosystem-sycl LANGUAGES CXX)

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
            "kamping-ecosystem-thrust: kamping::v2 target not found and the sibling monorepo "
            "directory '${_kamping_v2_local}' does not exist. "
            "Provide kamping::v2 before including this component, or build from the monorepo root."
        )
    endif ()
    FetchContent_Declare(kamping_v2 SOURCE_DIR "${_kamping_v2_local}")
    unset(_kamping_v2_local)
    FetchContent_MakeAvailable(kamping_v2)
endif ()

# find_package(AdaptiveCpp REQUIRED)

add_library(kamping_ecosystem_sycl INTERFACE)
target_include_directories(kamping_ecosystem_sycl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kamping_ecosystem_sycl INTERFACE kamping::v2)
# add_sycl_to_target(TARGET kamping_ecosystem_sycl)
add_library(kamping::ecosystem::sycl ALIAS kamping_ecosystem_sycl)

if (PROJECT_IS_TOP_LEVEL OR KAMPING_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(examples)
endif ()
