cmake_minimum_required(VERSION 3.25)
project(kamping-v2-kokkos LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/kamping-v2-setup.cmake)
kamping_v2_require_component(kamping-v2 kamping::v2)

FetchContent_Declare(
    Kokkos
    URL https://github.com/kokkos/kokkos/releases/download/5.0.2/kokkos-5.0.2.tar.gz
    URL_HASH SHA256=188817bb452ca805ee8701f1c5adbbb4fb83dc8d1c50624566a18a719ba0fa5e
    OVERRIDE_FIND_PACKAGE SYSTEM
)
FetchContent_MakeAvailable(Kokkos)

kamping_v2_add_header_only_library(
    kamping_ecosystem_kokkos kamping::ecosystem::kokkos ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(kamping_ecosystem_kokkos PROPERTIES EXPORT_NAME kokkos)
target_link_libraries(kamping_ecosystem_kokkos INTERFACE kamping::v2 Kokkos::kokkos)
set(kokkos_dependencies "find_dependency(Kokkos)")

if (KAMPING_V2_ENABLE_KOKKOS_COMM)
    FetchContent_Declare(
        KokkosComm
        GIT_REPOSITORY https://github.com/kamping-site/kokkos-comm.git
        GIT_TAG v0.0.2
        SYSTEM
    )
    FetchContent_MakeAvailable(KokkosComm)
    target_compile_definitions(kamping_ecosystem_kokkos INTERFACE KAMPING_HAS_KOKKOS_COMM)
    target_link_libraries(kamping_ecosystem_kokkos INTERFACE KokkosComm::KokkosComm)
    list(APPEND kokkos_dependencies "find_dependency(KokkosComm)")
endif ()

kamping_v2_install_component(
    kokkos
    TARGETS
    kamping_ecosystem_kokkos
    NAMESPACE
    kamping::ecosystem::
    REQUIRES
    v2
    DEPENDENCIES
    ${kokkos_dependencies}
)

if (PROJECT_IS_TOP_LEVEL OR KAMPING_V2_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(examples)
    add_subdirectory(tests)
endif ()
