cmake_minimum_required(VERSION 3.25)
project(kamping-v2 LANGUAGES CXX)

include(FetchContent)

# When kamping-v2 is used as the FetchContent SOURCE_SUBDIR entry point (i.e. the monorepo
# root CMakeLists.txt was not processed), bootstrap all required dependencies here.
# In the monorepo build these are already provided by the root CMakeLists.txt.
if (NOT TARGET mpi::core)
    set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
    find_package(MPI REQUIRED)

    FetchContent_Declare(
        kamping_kassert
        GIT_REPOSITORY https://github.com/kamping-site/kassert
        GIT_TAG v1.0.0
    )
    set(KASSERT_VENDOR_ID kamping)
    set(KASSERT_CXX_NAMESPACE kamping::kassert)
    set(KASSERT_INCLUDE_SUBDIR kamping/kassert)
    set(KASSERT_PREFIX KAMPING_ASSERT)
    FetchContent_MakeAvailable(kamping_kassert)

    # Declare pfr so kamping-types can discover it (it calls FetchContent_MakeAvailable(pfr)
    # internally when Boost::pfr is not already a target).
    FetchContent_Declare(
        pfr
        GIT_REPOSITORY https://github.com/boostorg/pfr
        GIT_TAG 2.2.0
        SYSTEM FIND_PACKAGE_ARGS 2.2.0
    )

    # kamping-types lives in the kamping-site/kamping (v1) repo.
    FetchContent_Declare(
        kamping_types
        GIT_REPOSITORY https://github.com/kamping-site/kamping
        GIT_TAG main
        SOURCE_SUBDIR kamping-types
    )
    FetchContent_MakeAvailable(kamping_types)

    set(_mpi_core_local "${CMAKE_CURRENT_SOURCE_DIR}/../mpi-core")
    if (NOT EXISTS "${_mpi_core_local}/CMakeLists.txt")
        message(
            FATAL_ERROR
            "kamping-v2: mpi::core target not found and the sibling directory "
            "'${_mpi_core_local}' does not exist. Provide mpi::core before including "
            "this component, or build from the monorepo root."
        )
    endif ()
    FetchContent_Declare(mpi_core SOURCE_DIR "${_mpi_core_local}")
    unset(_mpi_core_local)
    FetchContent_MakeAvailable(mpi_core)
endif ()

add_library(kamping_v2 INTERFACE)
target_include_directories(kamping_v2 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kamping_v2 INTERFACE cxx_std_20)
target_link_libraries(kamping_v2 INTERFACE mpi::core)
add_library(kamping::v2 ALIAS kamping_v2)

# Interface target for warning flags. The monorepo root or any downstream consumer can append
# additional flags (e.g. -Werror) to this target after processing this file.
add_library(kamping_v2_warnings INTERFACE)
target_compile_options(
    kamping_v2_warnings INTERFACE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
add_library(kamping::v2::warnings ALIAS kamping_v2_warnings)

if (PROJECT_IS_TOP_LEVEL OR KAMPING_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(tests)
    add_subdirectory(examples)
endif ()
