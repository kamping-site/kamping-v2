# Installs this build tree into a scratch prefix and builds a consumer project against the installed package, to check
# that find_package(kamping-v2) works.

set(prefix "${CMAKE_CURRENT_BINARY_DIR}/prefix")
set(consumer_build "${CMAKE_CURRENT_BINARY_DIR}/consumer")

add_test(NAME test_install.install COMMAND ${CMAKE_COMMAND} --install ${PROJECT_BINARY_DIR} --prefix ${prefix} --config
                                           $<CONFIG>
)
add_test(
    NAME test_install.configure
    COMMAND
        ${CMAKE_COMMAND} -S ${CMAKE_CURRENT_SOURCE_DIR}/consumer -B ${consumer_build} -G ${CMAKE_GENERATOR}
        -DCMAKE_BUILD_TYPE=$<CONFIG> -DCMAKE_PREFIX_PATH=${prefix} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DMPI_CXX_COMPILER=${MPI_CXX_COMPILER}
)
add_test(NAME test_install.build COMMAND ${CMAKE_COMMAND} --build ${consumer_build} --config $<CONFIG>)
add_test(NAME test_install.run COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                                       ${consumer_build}/consumer ${MPIEXEC_POSTFLAGS}
)

set_tests_properties(test_install.install PROPERTIES FIXTURES_SETUP install_prefix)
set_tests_properties(
    test_install.configure PROPERTIES FIXTURES_REQUIRED install_prefix FIXTURES_SETUP install_consumer_configured
)
set_tests_properties(
    test_install.build PROPERTIES FIXTURES_REQUIRED install_consumer_configured FIXTURES_SETUP install_consumer_built
)
set_tests_properties(test_install.run PROPERTIES FIXTURES_REQUIRED install_consumer_built)
