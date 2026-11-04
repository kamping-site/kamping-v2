cmake_minimum_required(VERSION 3.25)
project(kamping-v2-kokkos LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/kamping-v2-setup.cmake)
if (NOT TARGET kamping::v2)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../kamping-v2 kamping-v2)
endif ()

FetchContent_Declare(
    Kokkos
    URL https://github.com/kokkos/kokkos/releases/download/5.0.2/kokkos-5.0.2.tar.gz
    URL_HASH SHA256=188817bb452ca805ee8701f1c5adbbb4fb83dc8d1c50624566a18a719ba0fa5e
    OVERRIDE_FIND_PACKAGE SYSTEM
)
FetchContent_MakeAvailable(Kokkos)

add_library(kamping_v2_kokkos INTERFACE)
add_library(kamping::v2::kokkos ALIAS kamping_v2_kokkos)
target_compile_features(kamping_v2_kokkos INTERFACE cxx_std_20)
target_sources(
    kamping_v2_kokkos INTERFACE FILE_SET HEADERS BASE_DIRS include FILES include/kamping/v2/ecosystem/kokkos.hpp
)
set_target_properties(kamping_v2_kokkos PROPERTIES EXPORT_NAME kokkos)
target_link_libraries(kamping_v2_kokkos INTERFACE kamping::v2 Kokkos::kokkos)

if (KAMPING_V2_ENABLE_KOKKOS_COMM)
    FetchContent_Declare(
        KokkosComm
        GIT_REPOSITORY https://github.com/kokkos/kokkos-comm.git
        GIT_TAG v0.1.0
        SYSTEM
    )
    block()
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(KokkosComm_ENABLE_TESTS OFF)
    set(KokkosComm_ENABLE_PERFTESTS OFF)
    # Installed alongside kamping-v2, whose kokkos component finds it with find_dependency().
    set(KokkosComm_INSTALL_CMAKE_PACKAGE ${KAMPING_V2_INSTALL})
    FetchContent_MakeAvailable(KokkosComm)
    endblock()
    # KokkosComm v0.1.0 generates config.hpp below its PROJECT_BINARY_DIR, but lists it in its header file set below
    # CMAKE_BINARY_DIR. These differ when it is not the top-level project, which breaks installing it; copy the header
    # to where the file set expects it.
    FetchContent_GetProperties(KokkosComm BINARY_DIR kokkoscomm_binary_dir)
    file(COPY "${kokkoscomm_binary_dir}/src/KokkosComm/config.hpp" DESTINATION "${CMAKE_BINARY_DIR}/src/KokkosComm")
    # KokkosComm only provides the namespaced name in its installed export; add it for the build tree, too.
    if (NOT TARGET KokkosComm::KokkosComm)
        add_library(KokkosComm::KokkosComm ALIAS KokkosComm)
    endif ()
    target_compile_definitions(kamping_v2_kokkos INTERFACE KAMPING_HAS_KOKKOS_COMM)
    target_link_libraries(kamping_v2_kokkos INTERFACE KokkosComm::KokkosComm)
endif ()

if (KAMPING_V2_INSTALL)
    install(
        TARGETS kamping_v2_kokkos
        EXPORT kamping-v2-kokkos-targets
        FILE_SET HEADERS
    )
    install(
        EXPORT kamping-v2-kokkos-targets
        NAMESPACE kamping::v2::
        DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR}
    )
    configure_file(cmake/kamping-v2-kokkos.cmake.in kamping-v2-kokkos.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/kamping-v2-kokkos.cmake DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR})
endif ()

if (PROJECT_IS_TOP_LEVEL OR KAMPING_V2_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(examples)
    add_subdirectory(tests)
endif ()
