cmake_minimum_required(VERSION 3.25)
project(mpi-core LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/kamping-v2-setup.cmake)

kamping_v2_add_header_only_library(mpi_core mpi::core ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(mpi_core PROPERTIES EXPORT_NAME core)
target_link_libraries(mpi_core INTERFACE MPI::MPI_CXX kamping_v2::kassert kamping::types)

# Default assertion levels, so kassert does not warn about a missing definition. Debug: normal assertions (level 30, the
# level of a plain KAMPING_V2_ASSERT(...)); otherwise: exceptions only (level 10).
target_compile_definitions(
    mpi_core INTERFACE $<IF:$<CONFIG:Debug>,KAMPING_V2_ASSERT_ASSERTION_LEVEL=30,KAMPING_V2_ASSERT_ASSERTION_LEVEL=10>
)
# kamping-types uses its own kassert instance (macro prefix KAMPING_ASSERT); set its level too.
target_compile_definitions(
    mpi_core INTERFACE $<IF:$<CONFIG:Debug>,KAMPING_ASSERT_ASSERTION_LEVEL=20,KAMPING_ASSERT_ASSERTION_LEVEL=10>
)

kamping_v2_install_component(
    core
    TARGETS
    mpi_core
    NAMESPACE
    mpi::
    REQUIRES
    types
    DEPENDENCIES
    "find_dependency(MPI COMPONENTS CXX)"
    "find_dependency(kamping_v2 CONFIG PATHS \"\${CMAKE_CURRENT_LIST_DIR}/../kamping/v2\" NO_DEFAULT_PATH)"
)

if (PROJECT_IS_TOP_LEVEL OR KAMPING_V2_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(tests)
endif ()
