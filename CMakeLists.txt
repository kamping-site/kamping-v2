cmake_minimum_required(VERSION 3.25)

include(cmake/kamping-v2-version.cmake)
project(
    kamping-v2
    DESCRIPTION "KaMPIng v2: concept-based C++20 MPI bindings"
    LANGUAGES CXX
    VERSION ${KAMPING_V2_VERSION}
)

# Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if (EXISTS "${LOC_PATH}")
    message(
        FATAL_ERROR
            "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build "
            "subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles."
    )
endif ()

if (PROJECT_IS_TOP_LEVEL)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    include(FetchContent)
    FetchContent_Declare(
        Format.cmake
        GIT_REPOSITORY https://github.com/TheLartians/Format.cmake
        GIT_TAG v1.8.1
    )
    FetchContent_MakeAvailable(Format.cmake)
endif ()

include(cmake/kamping-v2-setup.cmake)

add_subdirectory(mpi-core)
add_subdirectory(kamping-v2)
add_subdirectory(dstl)
add_subdirectory(ecosystem)

if (PROJECT_IS_TOP_LEVEL
    AND BUILD_TESTING
    AND KAMPING_V2_BUILD_EXAMPLES_AND_TESTS
    AND KAMPING_V2_INSTALL
)
    add_subdirectory(tests/install)
endif ()
