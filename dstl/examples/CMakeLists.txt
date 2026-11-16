# Shows the OpenMP thread layout and uses execution_policy::par.
if (KAMPING_V2_ENABLE_OPENMP)
    add_executable(example_dstl_grid_visualization grid_visualization_example.cpp)
    target_link_libraries(example_dstl_grid_visualization PRIVATE dstl::dstl kamping_v2_warnings)
    target_compile_features(example_dstl_grid_visualization PRIVATE cxx_std_20)
endif ()
