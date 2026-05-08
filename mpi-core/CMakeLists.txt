cmake_minimum_required(VERSION 3.25)
project(mpi-core LANGUAGES CXX)

include(FetchContent)

# Bootstrap dependencies when mpi-core is used as a standalone FetchContent entry point.
# In the monorepo build these targets are already provided by the root CMakeLists.txt.
if (NOT TARGET MPI::MPI_CXX)
    set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
    find_package(MPI REQUIRED)
endif ()

if (NOT TARGET kamping::kassert)
    FetchContent_Declare(
        kamping_v2_kassert
        GIT_REPOSITORY https://github.com/kamping-site/kassert
        GIT_TAG v1.0.0
    )
    set(KASSERT_VENDOR_ID kamping)
    set(KASSERT_CXX_NAMESPACE kamping::kassert)
    set(KASSERT_INCLUDE_SUBDIR kamping/kassert)
    set(KASSERT_PREFIX KAMPING_ASSERT)
    FetchContent_MakeAvailable(kamping_v2_kassert)
endif ()

if (NOT TARGET kamping::types)
    FetchContent_Declare(
        pfr
        GIT_REPOSITORY https://github.com/boostorg/pfr
        GIT_TAG 2.2.0
        SYSTEM FIND_PACKAGE_ARGS 2.2.0
    )
    FetchContent_Declare(
        kamping_types
        GIT_REPOSITORY https://github.com/kamping-site/kamping
        GIT_TAG main
        SOURCE_SUBDIR kamping-types
    )
    FetchContent_MakeAvailable(kamping_types)
endif ()

add_library(mpi_core INTERFACE)
target_include_directories(mpi_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(mpi_core INTERFACE cxx_std_20)
target_link_libraries(mpi_core INTERFACE MPI::MPI_CXX kamping::kassert kamping::types)

# Set a sensible default assertion level so kassert does not warn about a missing definition
# when mpi-core is used without kamping-v1 (which sets its own level on the kamping target).
# Debug → light assertions (level 20); otherwise → exceptions only (level 10).
target_compile_definitions(
    mpi_core INTERFACE
    $<IF:$<CONFIG:Debug>,KAMPING_ASSERT_ASSERTION_LEVEL=20,KAMPING_ASSERT_ASSERTION_LEVEL=10>
)

add_library(mpi::core ALIAS mpi_core)
