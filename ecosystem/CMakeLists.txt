cmake_minimum_required(VERSION 3.25)
project(kamping-ecosystem LANGUAGES CXX)

option(KAMPING_ENABLE_SERIALIZATION "Enable the Cereal serialization adapter for kamping-v2." OFF)
option(KAMPING_ENABLE_THRUST "Enable the Thrust device_vector adapter for kamping-v2." OFF)
option(KAMPING_ENABLE_SYCL "Enable the SYCL buffer adapter for kamping-v2." OFF)

if (KAMPING_ENABLE_SERIALIZATION)
    add_subdirectory(cereal)
endif ()

if (KAMPING_ENABLE_THRUST)
    add_subdirectory(thrust)
endif ()

option(KAMPING_ENABLE_KOKKOS "Enable the Kokkos adapter for kamping-v2." OFF)
option(KAMPING_DISABLE_KOKKOS_COMM "Disable the KokkosComm dependency for the kokkos_view" ON)

if (KAMPING_ENABLE_KOKKOS)
    add_subdirectory(kokkos)
  endif ()
if (KAMPING_ENABLE_SYCL)
   add_subdirectory(sycl)
endif()
