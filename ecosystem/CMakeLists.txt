# Adapters for external libraries, each enabled by its KAMPING_V2_ENABLE_<NAME> option (see
# cmake/kamping-v2-setup.cmake).

if (KAMPING_V2_ENABLE_CEREAL)
    add_subdirectory(cereal)
endif ()
if (KAMPING_V2_ENABLE_KOKKOS)
    add_subdirectory(kokkos)
endif ()
if (KAMPING_V2_ENABLE_THRUST)
    add_subdirectory(thrust)
endif ()
if (KAMPING_V2_ENABLE_SYCL)
    add_subdirectory(sycl)
endif ()
