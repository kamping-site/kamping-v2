cmake_minimum_required(VERSION 3.25)
project(mpi-core LANGUAGES CXX)

add_library(mpi_core INTERFACE)
target_include_directories(mpi_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(mpi_core INTERFACE cxx_std_23)
target_link_libraries(mpi_core INTERFACE MPI::MPI_CXX kamping::kassert kamping::types)

# Set a sensible default assertion level so kassert does not warn about a missing definition
# when mpi-core is used without kamping-v1 (which sets its own level on the kamping target).
# Debug → light assertions (level 20); otherwise → exceptions only (level 10).
target_compile_definitions(
    mpi_core INTERFACE
    $<IF:$<CONFIG:Debug>,KAMPING_ASSERT_ASSERTION_LEVEL=20,KAMPING_ASSERT_ASSERTION_LEVEL=10>
)

add_library(kamping::mpi_core ALIAS mpi_core)
