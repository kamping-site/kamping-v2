include(GoogleTest)

FetchContent_MakeAvailable(KaTestrophe)

# Views and buffer protocol (no MPI communication)
add_executable(test_v2_auto_counts_view auto_counts_view_test.cpp)
target_link_libraries(test_v2_auto_counts_view PRIVATE kamping::v2 kamping_v2_warnings GTest::gtest_main GTest::gmock)
gtest_discover_tests(test_v2_auto_counts_view)

add_executable(test_v2_chunked_view chunked_view_test.cpp)
target_link_libraries(test_v2_chunked_view PRIVATE kamping::v2 kamping_v2_warnings GTest::gtest_main GTest::gmock)
gtest_discover_tests(test_v2_chunked_view)

add_executable(test_v2_counts_displs_by_reference counts_displs_by_reference_test.cpp)
target_link_libraries(
    test_v2_counts_displs_by_reference PRIVATE kamping::v2 kamping_v2_warnings GTest::gtest_main GTest::gmock
)
gtest_discover_tests(test_v2_counts_displs_by_reference)

add_executable(test_v2_flatten_v_view flatten_v_view_test.cpp)
target_link_libraries(test_v2_flatten_v_view PRIVATE kamping::v2 kamping_v2_warnings GTest::gtest_main GTest::gmock)
gtest_discover_tests(test_v2_flatten_v_view)

add_executable(test_v2_materialize materialize_test.cpp)
target_link_libraries(test_v2_materialize PRIVATE kamping::v2 kamping_v2_warnings GTest::gtest_main GTest::gmock)
gtest_discover_tests(test_v2_materialize)

add_executable(test_v2_resize_contract resize_contract_test.cpp)
target_link_libraries(test_v2_resize_contract PRIVATE kamping::v2 kamping_v2_warnings GTest::gtest_main GTest::gmock)
gtest_discover_tests(test_v2_resize_contract)

add_executable(test_v2_sentinels sentinels_test.cpp)
target_link_libraries(test_v2_sentinels PRIVATE kamping::v2 kamping_v2_warnings GTest::gtest_main GTest::gmock)
gtest_discover_tests(test_v2_sentinels)

add_executable(test_v2_umbrella_header umbrella_header_test.cpp)
target_link_libraries(test_v2_umbrella_header PRIVATE kamping::v2 kamping_v2_warnings GTest::gtest_main GTest::gmock)
gtest_discover_tests(test_v2_umbrella_header)

add_executable(test_v2_view_base_rvalue view_base_rvalue_test.cpp)
target_link_libraries(test_v2_view_base_rvalue PRIVATE kamping::v2 kamping_v2_warnings GTest::gtest_main GTest::gmock)
gtest_discover_tests(test_v2_view_base_rvalue)

add_executable(test_v2_view_interface_underlying view_interface_underlying_test.cpp)
target_link_libraries(
    test_v2_view_interface_underlying PRIVATE kamping::v2 kamping_v2_warnings GTest::gtest_main GTest::gmock
)
gtest_discover_tests(test_v2_view_interface_underlying)

# Datatypes and type pools
add_executable(test_v2_comm_view_with_pool comm_view_with_pool_test.cpp)
target_link_libraries(test_v2_comm_view_with_pool PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_comm_view_with_pool CORES 1 2)

add_executable(test_v2_type_pool type_pool_test.cpp)
target_link_libraries(test_v2_type_pool PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_type_pool CORES 1 2)

add_executable(test_v2_value_type value_type_test.cpp)
target_link_libraries(test_v2_value_type PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_value_type CORES 1 2 4)

# Requests, completion and non-blocking results
add_executable(test_v2_ibarrier ibarrier_test.cpp)
target_link_libraries(test_v2_ibarrier PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_ibarrier CORES 1 2 4)

add_executable(test_v2_iresult iresult_test.cpp)
target_link_libraries(test_v2_iresult PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_iresult CORES 1)

# Dropping a still-active request must call std::terminate. Each scenario runs as its own process, since it ends it.
# Covers both layers: the core request and request_vector, and the v2 iresult that builds on them.
add_executable(test_v2_terminate_on_active_request terminate_on_active_request_test.cpp)
target_link_libraries(test_v2_terminate_on_active_request PRIVATE kamping::v2 kamping_v2_warnings)
foreach (
    scenario
    request_destructor
    request_move_assignment
    request_vector_destructor
    request_vector_clear
    request_vector_resize
    request_vector_pop_back
    iresult_destructor
    iresult_during_unwinding
)
    add_test(NAME test_v2_terminate_on_active_request.${scenario}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
                     $<TARGET_FILE:test_v2_terminate_on_active_request> ${MPIEXEC_POSTFLAGS} ${scenario}
    )
    set_tests_properties(
        test_v2_terminate_on_active_request.${scenario}
        PROPERTIES PASS_REGULAR_EXPRESSION "dropped while a request is still active" FAIL_REGULAR_EXPRESSION
                   "NOT TERMINATED" TIMEOUT 30
    )
endforeach ()
set_tests_properties(
    test_v2_terminate_on_active_request.iresult_during_unwinding
    PROPERTIES PASS_REGULAR_EXPRESSION "still active \\(during exception unwinding\\)"
)
add_test(NAME test_v2_terminate_on_active_request.completed_requests
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
                 $<TARGET_FILE:test_v2_terminate_on_active_request> ${MPIEXEC_POSTFLAGS} completed_requests
)
set_tests_properties(
    test_v2_terminate_on_active_request.completed_requests
    PROPERTIES PASS_REGULAR_EXPRESSION "CONTROL OK" FAIL_REGULAR_EXPRESSION "dropped while a request is still active"
               TIMEOUT 30
)

# Overload sets and default arguments of all operations
add_executable(test_v2_overloads overloads_test.cpp)
target_link_libraries(test_v2_overloads PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_overloads CORES 1 2 4)

# Point-to-point
add_executable(test_v2_isendrecv isendrecv_test.cpp)
target_link_libraries(test_v2_isendrecv PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_isendrecv CORES 1 2 4)

add_executable(test_v2_sendrecv_replace sendrecv_replace_test.cpp)
target_link_libraries(test_v2_sendrecv_replace PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_sendrecv_replace CORES 1 2 4)

add_executable(test_v2_mrecv mrecv_test.cpp)
target_link_libraries(test_v2_mrecv PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_mrecv CORES 2 4)

add_executable(test_v2_probe probe_test.cpp)
target_link_libraries(test_v2_probe PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_probe CORES 2)

add_executable(test_v2_recv recv_test.cpp)
target_link_libraries(test_v2_recv PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_recv CORES 2 4)

# Collectives
add_executable(test_v2_allgather allgather_test.cpp)
target_link_libraries(test_v2_allgather PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_allgather CORES 1 2 4)

add_executable(test_v2_allgatherv allgatherv_test.cpp)
target_link_libraries(test_v2_allgatherv PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_allgatherv CORES 1 2 4)

add_executable(test_v2_allreduce allreduce_test.cpp)
target_link_libraries(test_v2_allreduce PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_allreduce CORES 1 2 4)

add_executable(test_v2_alltoall alltoall_test.cpp)
target_link_libraries(test_v2_alltoall PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_alltoall CORES 1 2 4)

add_executable(test_v2_alltoallv alltoallv_test.cpp)
target_link_libraries(test_v2_alltoallv PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_alltoallv CORES 1 2 4)

add_executable(test_v2_exscan exscan_test.cpp)
target_link_libraries(test_v2_exscan PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_exscan CORES 1 2 4)

add_executable(test_v2_gather gather_test.cpp)
target_link_libraries(test_v2_gather PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_gather CORES 1 2 4)

add_executable(test_v2_gatherv gatherv_test.cpp)
target_link_libraries(test_v2_gatherv PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_gatherv CORES 1 2 4)

add_executable(test_v2_make_op make_op_test.cpp)
target_link_libraries(test_v2_make_op PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_make_op CORES 1 2 4)

add_executable(test_v2_reduce reduce_test.cpp)
target_link_libraries(test_v2_reduce PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_reduce CORES 1 2 4)

add_executable(test_v2_scan scan_test.cpp)
target_link_libraries(test_v2_scan PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_scan CORES 1 2 4)

add_executable(test_v2_scatter scatter_test.cpp)
target_link_libraries(test_v2_scatter PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_scatter CORES 1 2 4)

add_executable(test_v2_scatterv scatterv_test.cpp)
target_link_libraries(test_v2_scatterv PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
katestrophe_add_mpi_test(test_v2_scatterv CORES 1 2 4)

# Sessions (MPI 4.0)
if (MPI_CXX_VERSION VERSION_GREATER_EQUAL "4.0")
    add_executable(test_v2_session session_test.cpp)
    target_link_libraries(test_v2_session PRIVATE kamping::v2 kamping_v2_warnings KaTestrophe::main)
    katestrophe_add_mpi_test(test_v2_session CORES 1 2)
endif ()
