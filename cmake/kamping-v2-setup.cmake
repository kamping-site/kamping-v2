# Shared setup for all entry points of this repository: the root CMakeLists.txt as well as mpi-core/, kamping-v2/, dstl/
# and ecosystem/*/, which can each be used on their own (e.g. via FetchContent with SOURCE_SUBDIR). Include it after
# project(); it runs once per build tree.
#
# Provides - the KAMPING_V2_* options, - the dependencies MPI, kassert, Boost.PFR and kamping-types, - the
# kamping_v2_warnings target used by tests and examples, - kamping_v2_require_component() to pull in a sibling component
# that is not built yet, - kamping_v2_install_component() to install a component into the kamping-v2 CMake package.

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/kamping-v2-version.cmake)
include(CMakePackageConfigHelpers)
include(FetchContent)
include(GNUInstallDirs)

set(KAMPING_V2_SOURCE_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/.."
    CACHE INTERNAL ""
)
cmake_path(NORMAL_PATH KAMPING_V2_SOURCE_ROOT)

# ── Options
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────

option(KAMPING_V2_BUILD_EXAMPLES_AND_TESTS "Build the tests and examples of all components." OFF)
option(KAMPING_V2_WARNINGS_ARE_ERRORS "Treat compiler warnings as errors in tests and examples." OFF)
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

# ── Dependencies
# ───────────────────────────────────────────────────────────────────────────────────────────────────────

if (NOT TARGET MPI::MPI_CXX)
    set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
    find_package(MPI REQUIRED)
endif ()

# kassert, configured as an instance of its own (macro prefix KAMPING_V2_ASSERT, header kamping/v2/kassert.hpp). The
# variables stay inside the block() so that they do not leak into the kassert instance kamping-types configures.
if (NOT TARGET kamping_v2::kassert)
    block()
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(KASSERT_CMAKE_NAMESPACE kamping_v2)
    set(KASSERT_CXX_NAMESPACE kamping::v2::kassert)
    set(KASSERT_INCLUDE_SUBDIR kamping/v2)
    set(KASSERT_PREFIX KAMPING_V2_ASSERT)
    set(KASSERT_INSTALL ${KAMPING_V2_INSTALL})
    FetchContent_Declare(
        kamping_v2_kassert
        GIT_REPOSITORY https://github.com/kamping-site/kassert
        GIT_TAG v1.0.0
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
        GIT_TAG main
        SOURCE_SUBDIR kamping-types
    )
    FetchContent_MakeAvailable(kamping_types)
    endblock()
endif ()

# ── Warnings
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────

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

# ── Components
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────

# kamping_v2_require_component(<subdirectory> <target>)
#
# Adds the sibling component in <subdirectory> (relative to the repository root) unless <target> already exists, so that
# a component can be built on its own.
function (kamping_v2_require_component subdirectory target)
    if (NOT TARGET ${target})
        add_subdirectory("${KAMPING_V2_SOURCE_ROOT}/${subdirectory}" "${CMAKE_BINARY_DIR}/_kamping_v2/${subdirectory}")
    endif ()
endfunction ()

# kamping_v2_add_header_only_library(<target> <alias> <include directory>)
#
# Creates the INTERFACE library <target> with alias <alias>, whose headers are the *.hpp files below <include
# directory>, as a HEADERS file set (which also provides the include directory, in the build and install tree).
function (kamping_v2_add_header_only_library target alias include_dir)
    file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${include_dir}/*.hpp")
    add_library(${target} INTERFACE)
    target_sources(${target} INTERFACE FILE_SET HEADERS BASE_DIRS "${include_dir}" FILES ${headers})
    target_compile_features(${target} INTERFACE cxx_std_20)
    add_library(${alias} ALIAS ${target})
endfunction ()

set(KAMPING_V2_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/kamping-v2")

# kamping_v2_install_component( <component> TARGETS <target>...      the targets of the component NAMESPACE <namespace>
# namespace of the exported targets (e.g. "kamping::"; the name is the EXPORT_NAME) REQUIRES <component>...  other
# kamping-v2 components this one depends on DEPENDENCIES <code>...   find_dependency() calls for its external
# dependencies )
#
# Installs the targets and headers of a component and registers it in the kamping-v2 package, so that
# find_package(kamping-v2 COMPONENTS <component>) loads it together with its dependencies.
function (kamping_v2_install_component component)
    if (NOT KAMPING_V2_INSTALL)
        return()
    endif ()
    cmake_parse_arguments(ARG "" "NAMESPACE" "TARGETS;REQUIRES;DEPENDENCIES" ${ARGN})

    install(
        TARGETS ${ARG_TARGETS}
        EXPORT kamping-v2-${component}-targets
        FILE_SET HEADERS
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT ${component}
    )
    install(
        EXPORT kamping-v2-${component}-targets
        NAMESPACE ${ARG_NAMESPACE}
        DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR}
        COMPONENT ${component}
    )

    set(content "# Dependencies of the kamping-v2 component '${component}'. Generated by kamping-v2-setup.cmake.\n")
    string(APPEND content "set(_kamping_v2_requires_${component} ${ARG_REQUIRES})\n")
    foreach (dependency IN LISTS ARG_DEPENDENCIES)
        string(APPEND content "${dependency}\n")
    endforeach ()
    set(file "${CMAKE_BINARY_DIR}/kamping-v2-package/kamping-v2-${component}-dependencies.cmake")
    file(WRITE "${file}" "${content}")
    install(
        FILES "${file}"
        DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR}
        COMPONENT ${component}
    )

    set_property(GLOBAL APPEND PROPERTY KAMPING_V2_INSTALLED_COMPONENTS ${component})

    if ("types" IN_LIST ARG_REQUIRES)
        _kamping_v2_install_types()
    endif ()
endfunction ()

# Installs kamping-types (and Boost.PFR, if built from source) as the internal 'types' component: neither provides
# install rules of its own when fetched. Their targets are adapted here: the include directories they set point into the
# source tree, so they are replaced by a HEADERS file set.
function (_kamping_v2_install_types)
    get_property(done GLOBAL PROPERTY _KAMPING_V2_TYPES_INSTALLED)
    if (done)
        return()
    endif ()
    set_property(GLOBAL PROPERTY _KAMPING_V2_TYPES_INSTALLED TRUE)

    get_target_property(imported kamping_types IMPORTED)
    if (imported)
        message(FATAL_ERROR "kamping-v2: installing requires kamping-types to be built from source (FetchContent).")
    endif ()

    set(targets kamping_types)
    set(dependencies
        "find_dependency(MPI COMPONENTS CXX)"
        "find_dependency(kamping CONFIG PATHS \"\${CMAKE_CURRENT_LIST_DIR}/../kamping/kassert\" NO_DEFAULT_PATH)"
    )
    _kamping_v2_adopt_headers(kamping_types types)

    if (TARGET boost_pfr)
        get_target_property(pfr_imported boost_pfr IMPORTED)
        if (NOT pfr_imported)
            _kamping_v2_adopt_headers(boost_pfr pfr)
            list(APPEND targets boost_pfr)
        endif ()
    elseif (TARGET kamping_v2_pfr)
        list(APPEND targets kamping_v2_pfr)
        list(APPEND dependencies "find_dependency(Boost 1.75 COMPONENTS headers CONFIG)")
    elseif (TARGET Boost::pfr)
        list(APPEND dependencies "find_dependency(pfr 2.2.0)")
    endif ()

    kamping_v2_install_component(
        types
        TARGETS
        ${targets}
        NAMESPACE
        kamping_v2::
        DEPENDENCIES
        ${dependencies}
    )
endfunction ()

# Replaces the source-tree include directory of the fetched target <target> (the include/ directory next to the
# CMakeLists.txt that defines it) by a HEADERS file set.
function (_kamping_v2_adopt_headers target export_name)
    get_target_property(source_dir ${target} SOURCE_DIR)
    set(dir "${source_dir}/include")
    file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${dir}/*.hpp")
    set_property(TARGET ${target} PROPERTY INTERFACE_INCLUDE_DIRECTORIES "")
    target_sources(${target} INTERFACE FILE_SET HEADERS BASE_DIRS "${dir}" FILES ${headers})
    set_target_properties(${target} PROPERTIES EXPORT_NAME ${export_name})
endfunction ()

# Writes the package configuration once all components of this build tree have registered. Deferred to the end of the
# directory that included this file first, i.e. the entry point.
function (_kamping_v2_install_package)
    if (NOT KAMPING_V2_INSTALL)
        return()
    endif ()
    get_property(components GLOBAL PROPERTY KAMPING_V2_INSTALLED_COMPONENTS)
    set(KAMPING_V2_PACKAGE_COMPONENTS ${components})
    set(dir "${CMAKE_BINARY_DIR}/kamping-v2-package")
    configure_package_config_file(
        "${KAMPING_V2_SOURCE_ROOT}/cmake/kamping-v2-config.cmake.in" "${dir}/kamping-v2-config.cmake"
        INSTALL_DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR}
    )
    write_basic_package_version_file(
        "${dir}/kamping-v2-config-version.cmake"
        VERSION ${KAMPING_V2_VERSION}
        COMPATIBILITY SameMinorVersion ARCH_INDEPENDENT
    )
    install(FILES "${dir}/kamping-v2-config.cmake" "${dir}/kamping-v2-config-version.cmake"
            DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR}
    )
endfunction ()
cmake_language(DEFER DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" CALL _kamping_v2_install_package)
