cmake_minimum_required(VERSION 3.25)
project(kamping-ecosystem-kokkos LANGUAGES CXX)

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
                "kamping-ecosystem-cereal: kamping::v2 target not found and the sibling monorepo "
                "directory '${_kamping_v2_local}' does not exist. "
                "Provide kamping::v2 before including this component, or build from the monorepo root."
        )
    endif ()
    FetchContent_Declare(kamping_v2 SOURCE_DIR "${_kamping_v2_local}")
    unset(_kamping_v2_local)
    FetchContent_MakeAvailable(kamping_v2)
endif ()

# Declare Kokkos and KokkosComm here so this component is self-contained. If kamping-v1 (or the consumer)
# already made KoKKos/KokkosComm available, FetchContent_MakeAvailable is a no-op.
FetchContent_Declare(
        Kokkos
        URL https://github.com/kokkos/kokkos/releases/download/5.0.2/kokkos-5.0.2.tar.gz
        URL_HASH SHA256=188817bb452ca805ee8701f1c5adbbb4fb83dc8d1c50624566a18a719ba0fa5e
        OVERRIDE_FIND_PACKAGE
        SYSTEM
)
FetchContent_Declare(
        KokkosComm
        GIT_REPOSITORY https://github.com/kamping-site/kokkos-comm.git
        GIT_TAG develop
        SYSTEM
)

FetchContent_MakeAvailable(Kokkos KokkosComm)

add_library(kamping_ecosystem_kokkos INTERFACE)
target_include_directories(kamping_ecosystem_kokkos INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kamping_ecosystem_kokkos INTERFACE kamping::v2 Kokkos::kokkos KokkosComm::KokkosComm)
add_library(kamping::ecosystem::kokkos ALIAS kamping_ecosystem_cereal)

if (PROJECT_IS_TOP_LEVEL OR KAMPING_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(examples)
endif ()
