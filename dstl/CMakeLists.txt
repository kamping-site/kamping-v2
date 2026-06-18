cmake_minimum_required(VERSION 3.25)
project(dstl LANGUAGES CXX)

include(FetchContent)

# Bootstrap dependencies when dstl is used as a standalone FetchContent entry point.
# In the monorepo build, kamping::v2 (and transitively mpi::core) is provided by the root.
if (NOT TARGET kamping::v2)
    set(_kamping_v2_local "${CMAKE_CURRENT_SOURCE_DIR}/../kamping-v2")
    if (NOT EXISTS "${_kamping_v2_local}/CMakeLists.txt")
        message(
            FATAL_ERROR
            "dstl: kamping::v2 target not found and the sibling directory '${_kamping_v2_local}' does not exist. "
            "Provide kamping::v2 before including this component, or build from the monorepo root."
        )
    endif ()
    FetchContent_Declare(kamping_v2 SOURCE_DIR "${_kamping_v2_local}")
    FetchContent_MakeAvailable(kamping_v2)
    unset(_kamping_v2_local)
endif ()

add_library(dstl INTERFACE)
target_include_directories(dstl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(dstl INTERFACE cxx_std_20)
target_link_libraries(dstl INTERFACE kamping::v2)
add_library(dstl::dstl ALIAS dstl)

# Optional OpenMP acceleration for the funneled / concurrent re-bin kernels. When not found, those
# execution policies fall back to a serial rebin (correctness is unaffected).
find_package(OpenMP QUIET)
if (OpenMP_CXX_FOUND)
    target_link_libraries(dstl INTERFACE OpenMP::OpenMP_CXX)
endif ()

if (PROJECT_IS_TOP_LEVEL OR KAMPING_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(tests)
    add_subdirectory(examples)
endif ()
