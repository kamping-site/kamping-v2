cmake_minimum_required(VERSION 3.25)

project(
    KaMPIng
    DESCRIPTION "KaMPIng v2 — concept-based C++20 MPI bindings"
    LANGUAGES CXX
    VERSION 0.1.0
)
include(FetchContent)

if (PROJECT_IS_TOP_LEVEL)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    include(CTest)

    FetchContent_Declare(
        Format.cmake
        GIT_REPOSITORY https://github.com/TheLartians/Format.cmake
        GIT_TAG v1.8.1
    )
    FetchContent_MakeAvailable(Format.cmake)
endif ()

# Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if (EXISTS "${LOC_PATH}")
    message(
        FATAL_ERROR
            "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build "
            "subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles."
    )
endif ()

option(KAMPING_BUILD_EXAMPLES_AND_TESTS "Build tests and examples for all components." OFF)
option(KAMPING_WARNINGS_ARE_ERRORS "Treat compiler warnings as errors." OFF)
option(KAMPING_ENABLE_REFLECTION "Enable Boost.PFR struct reflection in kamping-types." ON)
option(
    KAMPING_REFLECTION_USE_SYSTEM_BOOST_FOR_PFR
    "Use Boost.PFR from a system-installed Boost instead of a standalone install or building from source."
    OFF
)

set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
find_package(MPI REQUIRED)

# Configure the kassert variables inside a block() so they stay scoped to this kassert instance and do
# not leak into the shared kamping-types component below, which configures its own (kamping::kassert)
# kassert instance. The created target is global and remains available outside the block.
block()
    set(KASSERT_CMAKE_NAMESPACE kamping_v2)
    set(KASSERT_CXX_NAMESPACE kamping::v2::kassert)
    set(KASSERT_INCLUDE_SUBDIR kamping/v2)
    set(KASSERT_PREFIX KAMPING_V2_ASSERT)
    FetchContent_Declare(
        kamping_v2_kassert
        GIT_REPOSITORY https://github.com/kamping-site/kassert
        GIT_TAG v1.0.0
    )
    FetchContent_MakeAvailable(kamping_v2_kassert)
endblock()

# Boost.PFR: declare here so kamping-types can discover it. The system-Boost alias must be
# created before FetchContent_MakeAvailable(kamping_types) so kamping-types sees an existing
# Boost::pfr target and skips its own FetchContent.
FetchContent_Declare(
    pfr
    GIT_REPOSITORY https://github.com/boostorg/pfr
    GIT_TAG 2.2.0
    SYSTEM FIND_PACKAGE_ARGS 2.2.0
)

if (KAMPING_ENABLE_REFLECTION AND KAMPING_REFLECTION_USE_SYSTEM_BOOST_FOR_PFR)
    find_package(Boost 1.75 COMPONENTS headers CONFIG)
    if (NOT Boost_FOUND)
        message(
            FATAL_ERROR
                "Boost.PFR: No compatible Boost version found. Use KAMPING_REFLECTION_USE_SYSTEM_BOOST_FOR_PFR=OFF to use standalone Boost.PFR."
        )
    else ()
        message(STATUS "Found Boost ${Boost_VERSION}: ${Boost_DIR}")
        message(STATUS "Using system Boost for Boost.PFR")
        add_library(kamping_pfr INTERFACE)
        target_link_libraries(kamping_pfr INTERFACE Boost::headers)
        add_library(Boost::pfr ALIAS kamping_pfr)
    endif ()
endif ()

if (NOT TARGET kamping::types)
  # kamping-types lives in the kamping-site/kamping (v1) repo.
  if (KAMPING_ENABLE_REFLECTION)
    set(KAMPING_TYPES_ENABLE_REFLECTION ON)
  endif ()
  if (PROJECT_IS_TOP_LEVEL OR KAMPING_BUILD_EXAMPLES_AND_TESTS)
    set(KAMPING_TYPES_BUILD_EXAMPLES ON CACHE BOOL "" FORCE)
  endif ()
  FetchContent_Declare(
    kamping_types
    GIT_REPOSITORY https://github.com/kamping-site/kamping
    GIT_TAG main
    SOURCE_SUBDIR kamping-types
  )
  FetchContent_MakeAvailable(kamping_types)
endif ()

# Declare KaTestrophe once here so every test subdirectory shares the same version.
# FetchContent_Declare uses "first declaration wins" semantics: the subdirectories only
# call FetchContent_MakeAvailable(KaTestrophe) and inherit these details.
FetchContent_Declare(
    KaTestrophe
    GIT_REPOSITORY https://github.com/kamping-site/KaTestrophe.cmake
    GIT_TAG a7d27ca
)

add_subdirectory(mpi-core)
add_subdirectory(kamping-v2)

# Propagate -Werror to the v2 warnings target when requested.
if (KAMPING_WARNINGS_ARE_ERRORS)
    target_compile_options(kamping_v2_warnings INTERFACE -Werror)
endif ()

add_subdirectory(dstl)
add_subdirectory(ecosystem)
