include(FetchContent)
include(GoogleTest)

FetchContent_Declare(
    KaTestrophe
    GIT_REPOSITORY https://github.com/kamping-site/KaTestrophe.cmake
    GIT_TAG v1.1.0
)
FetchContent_MakeAvailable(KaTestrophe)

# kamping_v2_add_unit_test(<name>): a GoogleTest test without MPI, built from <name>_test.cpp.
function (kamping_v2_add_unit_test name)
    add_executable(test_v2_${name} ${name}_test.cpp)
    target_link_libraries(test_v2_${name} PRIVATE kamping_v2 kamping_v2_warnings GTest::gtest_main GTest::gmock)
    gtest_discover_tests(test_v2_${name})
endfunction ()

# kamping_v2_add_mpi_test(<name> CORES <n>...): an MPI test built from <name>_test.cpp, run on each core count.
function (kamping_v2_add_mpi_test name)
    cmake_parse_arguments(ARG "" "" "CORES" ${ARGN})
    add_executable(test_v2_${name} ${name}_test.cpp)
    target_link_libraries(test_v2_${name} PRIVATE kamping_v2 kamping_v2_warnings KaTestrophe::main)
    katestrophe_add_mpi_test(test_v2_${name} CORES ${ARG_CORES})
endfunction ()

# Views and buffer protocol (no MPI communication)
kamping_v2_add_unit_test(auto_counts_view)
kamping_v2_add_unit_test(chunked_view)
kamping_v2_add_unit_test(counts_displs_by_reference)
kamping_v2_add_unit_test(flatten_v_view)
kamping_v2_add_unit_test(materialize)
kamping_v2_add_unit_test(resize_contract)
kamping_v2_add_unit_test(sentinels)
kamping_v2_add_unit_test(umbrella_header)
kamping_v2_add_unit_test(view_base_rvalue)
kamping_v2_add_unit_test(view_interface_underlying)

# Datatypes and type pools
kamping_v2_add_mpi_test(comm_view_with_pool CORES 1 2)
kamping_v2_add_mpi_test(type_pool CORES 1 2)
kamping_v2_add_mpi_test(value_type CORES 1 2 4)

# Requests, completion and non-blocking results
kamping_v2_add_mpi_test(completion CORES 1 2 4)
kamping_v2_add_mpi_test(handle_vector CORES 1 2)
kamping_v2_add_mpi_test(ibarrier CORES 1 2 4)
kamping_v2_add_mpi_test(iresult CORES 1)
kamping_v2_add_mpi_test(request_view CORES 1 2)

# Dropping a still-active request must call std::terminate. Each scenario runs as its own process, since it ends it.
add_executable(test_v2_terminate_on_active_request terminate_on_active_request_test.cpp)
target_link_libraries(test_v2_terminate_on_active_request PRIVATE kamping_v2 kamping_v2_warnings)
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
kamping_v2_add_mpi_test(overloads CORES 1 2 4)

# Point-to-point
kamping_v2_add_mpi_test(isendrecv CORES 1 2 4)
kamping_v2_add_mpi_test(mrecv CORES 2 4)
kamping_v2_add_mpi_test(probe CORES 2)
kamping_v2_add_mpi_test(recv CORES 2 4)

# Collectives
kamping_v2_add_mpi_test(allgather CORES 1 2 4)
kamping_v2_add_mpi_test(allgatherv CORES 1 2 4)
kamping_v2_add_mpi_test(allreduce CORES 1 2 4)
kamping_v2_add_mpi_test(alltoall CORES 1 2 4)
kamping_v2_add_mpi_test(alltoallv CORES 1 2 4)
kamping_v2_add_mpi_test(exscan CORES 1 2 4)
kamping_v2_add_mpi_test(gather CORES 1 2 4)
kamping_v2_add_mpi_test(gatherv CORES 1 2 4)
kamping_v2_add_mpi_test(make_op CORES 1 2 4)
kamping_v2_add_mpi_test(reduce CORES 1 2 4)
kamping_v2_add_mpi_test(scan CORES 1 2 4)
kamping_v2_add_mpi_test(scatter CORES 1 2 4)
kamping_v2_add_mpi_test(scatterv CORES 1 2 4)

# Sessions (MPI 4.0)
if (MPI_CXX_VERSION VERSION_GREATER_EQUAL "4.0")
    kamping_v2_add_mpi_test(session CORES 1 2)
endif ()
