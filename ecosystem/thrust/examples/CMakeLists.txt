enable_language(CUDA)
add_executable(example_thrust_send_recv thrust_send_recv_example.cu)
target_link_libraries(example_thrust_send_recv PRIVATE kamping_ecosystem_thrust)
set_target_properties(example_thrust_send_recv PROPERTIES
    CXX_STANDARD 20
    CUDA_STANDARD 20
    CUDA_SEPARABLE_COMPILATION ON
)

# TODO fix this
# Suppress a GCC 15 -Wchanges-meaning hit in mpi-core when forwarded to nvcc's host compiler:
# `group group() const` in comm_accessors triggers the warning because the member name
# shadows the class name.
target_compile_options(example_thrust_send_recv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-changes-meaning>
)
