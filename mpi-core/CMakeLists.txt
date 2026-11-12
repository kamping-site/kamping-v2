cmake_minimum_required(VERSION 3.25)
project(mpi-core LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/kamping-v2-setup.cmake)

add_library(mpi_core INTERFACE)
add_library(mpi::core ALIAS mpi_core)
target_compile_features(mpi_core INTERFACE cxx_std_20)
target_sources(
    mpi_core
    INTERFACE FILE_SET
              HEADERS
              BASE_DIRS
              include
              FILES
              include/mpi/arrays.hpp
              include/mpi/attribute.hpp
              include/mpi/buffer.hpp
              include/mpi/cache.hpp
              include/mpi/collectives/allgather.hpp
              include/mpi/collectives/allgatherv.hpp
              include/mpi/collectives/allreduce.hpp
              include/mpi/collectives/alltoall.hpp
              include/mpi/collectives/alltoallv.hpp
              include/mpi/collectives/barrier.hpp
              include/mpi/collectives/bcast.hpp
              include/mpi/collectives/exscan.hpp
              include/mpi/collectives/gather.hpp
              include/mpi/collectives/gatherv.hpp
              include/mpi/collectives/ibarrier.hpp
              include/mpi/collectives/reduce.hpp
              include/mpi/collectives/scan.hpp
              include/mpi/collectives/scatter.hpp
              include/mpi/collectives/scatterv.hpp
              include/mpi/comm.hpp
              include/mpi/completion/wait.hpp
              include/mpi/completion/waitall.hpp
              include/mpi/completion/waitany.hpp
              include/mpi/completion/waitsome.hpp
              include/mpi/detail/handle_iterator.hpp
              include/mpi/environment.hpp
              include/mpi/error.hpp
              include/mpi/group.hpp
              include/mpi/handle.hpp
              include/mpi/info.hpp
              include/mpi/message.hpp
              include/mpi/mpi_span.hpp
              include/mpi/ops.hpp
              include/mpi/p2p/improbe.hpp
              include/mpi/p2p/imrecv.hpp
              include/mpi/p2p/iprobe.hpp
              include/mpi/p2p/irecv.hpp
              include/mpi/p2p/isend.hpp
              include/mpi/p2p/isendrecv.hpp
              include/mpi/p2p/mprobe.hpp
              include/mpi/p2p/mrecv.hpp
              include/mpi/p2p/probe.hpp
              include/mpi/p2p/recv.hpp
              include/mpi/p2p/send.hpp
              include/mpi/p2p/sendrecv.hpp
              include/mpi/p2p/sendrecv_replace.hpp
              include/mpi/range_concepts.hpp
              include/mpi/request.hpp
              include/mpi/request_vector.hpp
              include/mpi/sentinels.hpp
              include/mpi/session.hpp
              include/mpi/status.hpp
              include/mpi/status_vector.hpp
              include/mpi/thread_level.hpp
              include/mpi/win.hpp
)
set_target_properties(mpi_core PROPERTIES EXPORT_NAME core)
target_link_libraries(mpi_core INTERFACE MPI::MPI_CXX kamping_v2::kassert kamping::types)

if (KAMPING_V2_INSTALL)
    install(
        TARGETS mpi_core
        EXPORT kamping-v2-core-targets
        FILE_SET HEADERS
    )
    install(
        EXPORT kamping-v2-core-targets
        NAMESPACE mpi::
        DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR}
    )
    install(FILES cmake/kamping-v2-core.cmake DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR})
endif ()

if (PROJECT_IS_TOP_LEVEL OR KAMPING_V2_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(tests)
endif ()
