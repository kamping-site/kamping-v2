# Shared setup for all entry points of this repository: the root CMakeLists.txt as well as mpi-core/, kamping-v2/, dstl/
# and ecosystem/*/, which can each be used on their own (e.g. via FetchContent with SOURCE_SUBDIR). Include it after
# project(); it runs once per build tree. It defines the KAMPING_V2_* options, makes the dependencies available, and
# defines the kamping_v2_warnings target for tests and examples. If KAMPING_V2_INSTALL is set, it installs the package
# configuration and the internal 'types' component; each component installs itself.

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/kamping-v2-version.cmake)
include(CMakePackageConfigHelpers)
include(FetchContent)
include(GNUInstallDirs)

# ── Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────

option(KAMPING_V2_BUILD_EXAMPLES_AND_TESTS "Build the tests and examples of all components." OFF)
option(KAMPING_V2_BUILD_DOCS "Add the docs target, which builds the documentation with Doxygen and Sphinx." OFF)
option(KAMPING_V2_WARNINGS_ARE_ERRORS
       "Treat compiler warnings as errors in tests and examples, and Sphinx warnings in the docs." OFF
)
option(KAMPING_V2_INSTALL "Generate install rules for the kamping-v2 CMake package." ${PROJECT_IS_TOP_LEVEL})
option(KAMPING_V2_ENABLE_REFLECTION "Derive MPI datatypes for plain structs via Boost.PFR (in kamping-types)." ON)
option(KAMPING_V2_REFLECTION_USE_SYSTEM_BOOST_FOR_PFR
       "Use Boost.PFR from a system-installed Boost instead of a standalone install or building from source." OFF
)
option(KAMPING_V2_ENABLE_CEREAL "Build the Cereal serialization adapter." OFF)
option(KAMPING_V2_ENABLE_KOKKOS "Build the Kokkos adapter." OFF)
option(KAMPING_V2_ENABLE_KOKKOS_COMM "Let the Kokkos adapter use KokkosComm." OFF)
option(KAMPING_V2_ENABLE_THRUST "Build the Thrust device_vector adapter (requires CUDA)." OFF)
option(KAMPING_V2_ENABLE_SYCL "Build the SYCL adapter (requires Intel oneAPI DPC++)." OFF)

# Enable testing for whichever directory is the entry point of the build.
if (PROJECT_IS_TOP_LEVEL)
    include(CTest)
endif ()

# ── Dependencies ──────────────────────────────────────────────────────────────────────────────────────────────────────

if (NOT TARGET MPI::MPI_CXX)
    set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
    find_package(MPI REQUIRED)
endif ()

# Assertion level of kamping-v2: 30 (normal assertions) in Debug builds, 10 (exceptions only) otherwise. It is the
# default level of the kamping_v2::kassert instance, so it applies to every target linking to kamping-v2; a single target
# can override it with the KAMPING_V2_ASSERT_ASSERTION_LEVEL property. kamping-types has its own kassert instance, whose
# level is set with KAMPING_TYPES_ASSERTION_LEVEL.
set(KAMPING_V2_ASSERTION_LEVEL
    $<IF:$<CONFIG:Debug>,30,10>
    CACHE STRING "Assertion level of kamping-v2 (a number, may be a generator expression)."
)

# kassert, configured as an instance of its own (macro prefix KAMPING_V2_ASSERT, header kamping/v2/kassert.hpp). The
# variables stay inside the block() so that they do not leak into the kassert instance kamping-types configures.
if (NOT TARGET kamping_v2::kassert)
    block()
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(KASSERT_DEFAULT_ASSERTION_LEVEL ${KAMPING_V2_ASSERTION_LEVEL})
    set(KASSERT_CMAKE_NAMESPACE kamping_v2)
    set(KASSERT_CXX_NAMESPACE kamping::v2::kassert)
    set(KASSERT_INCLUDE_SUBDIR kamping/v2)
    set(KASSERT_PREFIX KAMPING_V2_ASSERT)
    set(KASSERT_INSTALL ${KAMPING_V2_INSTALL})
    FetchContent_Declare(
        kamping_v2_kassert
        GIT_REPOSITORY https://github.com/kamping-site/kassert
        GIT_TAG v1.1.0
    )
    FetchContent_MakeAvailable(kamping_v2_kassert)
    endblock()
endif ()

# Boost.PFR, used by kamping-types for reflection. Declared here so that kamping-types finds it; the system-Boost alias
# must exist before kamping-types is made available, so that kamping-types skips its own FetchContent.
FetchContent_Declare(
    pfr
    GIT_REPOSITORY https://github.com/boostorg/pfr
    GIT_TAG 2.2.0
    SYSTEM FIND_PACKAGE_ARGS 2.2.0
)
if (KAMPING_V2_ENABLE_REFLECTION
    AND KAMPING_V2_REFLECTION_USE_SYSTEM_BOOST_FOR_PFR
    AND NOT TARGET Boost::pfr
)
    find_package(Boost 1.75 COMPONENTS headers CONFIG)
    if (NOT Boost_FOUND)
        message(
            FATAL_ERROR
                "Boost.PFR: no compatible Boost found. Set KAMPING_V2_REFLECTION_USE_SYSTEM_BOOST_FOR_PFR=OFF to use "
                "standalone Boost.PFR."
        )
    endif ()
    add_library(kamping_v2_pfr INTERFACE)
    target_link_libraries(kamping_v2_pfr INTERFACE Boost::headers)
    set_target_properties(kamping_v2_pfr PROPERTIES EXPORT_NAME pfr)
    add_library(Boost::pfr ALIAS kamping_v2_pfr)
endif ()

# KaTestrophe (GoogleTest with MPI), for the tests. Only declared here; the tests/ directories of the components fetch it
# with FetchContent_MakeAvailable(KaTestrophe), so it is downloaded only when tests are built.
FetchContent_Declare(
    KaTestrophe
    GIT_REPOSITORY https://github.com/kamping-site/KaTestrophe.cmake
    GIT_TAG v1.1.0
)

# kamping-types (MPI datatype construction), maintained in the KaMPIng (v1) repository.
if (NOT TARGET kamping::types)
    block()
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(KAMPING_TYPES_ENABLE_REFLECTION ${KAMPING_V2_ENABLE_REFLECTION})
    set(KAMPING_TYPES_BUILD_EXAMPLES OFF)
    set(KASSERT_INSTALL ${KAMPING_V2_INSTALL})
    FetchContent_Declare(
        kamping_types
        GIT_REPOSITORY https://github.com/kamping-site/kamping
        GIT_TAG v0.3.2
        SOURCE_SUBDIR kamping-types
    )
    FetchContent_MakeAvailable(kamping_types)
    endblock()
endif ()

# ── Warnings ──────────────────────────────────────────────────────────────────────────────────────────────────────────

# Warning flags for the tests and examples of all components. Not part of the installed package.
add_library(kamping_v2_warnings INTERFACE)
target_compile_options(
    kamping_v2_warnings
    INTERFACE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall
              -Wextra
              -Wpedantic
              -Wshadow
              -Wconversion
              -Wsign-conversion>
              $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
if (KAMPING_V2_WARNINGS_ARE_ERRORS)
    target_compile_options(kamping_v2_warnings INTERFACE $<IF:$<CXX_COMPILER_ID:MSVC>,/WX,-Werror>)
endif ()

# ── Package ───────────────────────────────────────────────────────────────────────────────────────────────────────────

# Every component <c> installs its targets as kamping-v2-<c>-targets.cmake and a file kamping-v2-<c>.cmake that finds
# the dependencies of <c> and includes its targets; kamping-v2-config.cmake loads the requested components.
set(KAMPING_V2_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/kamping-v2")

if (KAMPING_V2_INSTALL)
    configure_package_config_file(
        "${CMAKE_CURRENT_LIST_DIR}/kamping-v2-config.cmake.in" "${CMAKE_BINARY_DIR}/kamping-v2-config.cmake"
        INSTALL_DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR}
    )
    write_basic_package_version_file(
        "${CMAKE_BINARY_DIR}/kamping-v2-config-version.cmake"
        VERSION ${KAMPING_V2_VERSION}
        COMPATIBILITY SameMinorVersion ARCH_INDEPENDENT
    )
    install(FILES "${CMAKE_BINARY_DIR}/kamping-v2-config.cmake" "${CMAKE_BINARY_DIR}/kamping-v2-config-version.cmake"
            DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR}
    )

    block()
    # The internal 'types' component: kamping-types, and Boost.PFR if it is built from source or wrapped from a system
    # Boost. Neither provides install rules when fetched, and their include directories point into the source tree.
    get_target_property(imported kamping_types IMPORTED)
    if (imported)
        message(FATAL_ERROR "kamping-v2: installing requires kamping-types to be built from source (FetchContent).")
    endif ()
    set_target_properties(kamping_types PROPERTIES EXPORT_NAME types)
    set(types_targets kamping_types)
    set(vendored kamping_types)
    set(KAMPING_V2_PFR_DEPENDENCY "")
    if (TARGET boost_pfr)
        set_target_properties(boost_pfr PROPERTIES EXPORT_NAME pfr)
        list(APPEND types_targets boost_pfr)
        list(APPEND vendored boost_pfr)
    elseif (TARGET kamping_v2_pfr)
        list(APPEND types_targets kamping_v2_pfr)
        set(KAMPING_V2_PFR_DEPENDENCY "find_dependency(Boost 1.75 COMPONENTS headers CONFIG)")
    elseif (TARGET Boost::pfr)
        set(KAMPING_V2_PFR_DEPENDENCY "find_dependency(pfr 2.2.0)")
    endif ()
    foreach (target IN LISTS vendored)
        get_target_property(source_dir ${target} SOURCE_DIR)
        set_target_properties(
            ${target}
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                       "$<BUILD_INTERFACE:${source_dir}/include>;$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        )
        install(DIRECTORY "${source_dir}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    endforeach ()
    install(TARGETS ${types_targets} EXPORT kamping-v2-types-targets)
    install(
        EXPORT kamping-v2-types-targets
        NAMESPACE kamping_v2::
        DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR}
    )
    configure_file(
        "${CMAKE_CURRENT_LIST_DIR}/kamping-v2-types.cmake.in" "${CMAKE_BINARY_DIR}/kamping-v2-types.cmake" @ONLY
    )
    install(FILES "${CMAKE_BINARY_DIR}/kamping-v2-types.cmake" DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR})
    endblock()
endif ()
