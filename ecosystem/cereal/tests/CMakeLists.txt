include(FetchContent)

FetchContent_Declare(
    KaTestrophe
    GIT_REPOSITORY https://github.com/kamping-site/KaTestrophe.cmake
    GIT_TAG v1.0.2
)
FetchContent_MakeAvailable(KaTestrophe)

add_executable(test_cereal_view cereal_view_test.cpp)
target_link_libraries(test_cereal_view PRIVATE kamping_ecosystem_cereal kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_cereal_view CORES 2 4)
