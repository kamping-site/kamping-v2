cmake_minimum_required(VERSION 3.25)
project(kamping-v2-bindings LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/kamping-v2-setup.cmake)
if (NOT TARGET mpi::core)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../mpi-core mpi-core)
endif ()

add_library(kamping_v2 INTERFACE)
add_library(kamping::v2 ALIAS kamping_v2)
target_compile_features(kamping_v2 INTERFACE cxx_std_20)
target_sources(
    kamping_v2
    INTERFACE FILE_SET
              HEADERS
              BASE_DIRS
              include
              FILES
              include/kamping/v2.hpp
              include/kamping/v2/buffer.hpp
              include/kamping/v2/cache.hpp
              include/kamping/v2/collectives.hpp
              include/kamping/v2/collectives/allgather.hpp
              include/kamping/v2/collectives/allgatherv.hpp
              include/kamping/v2/collectives/allreduce.hpp
              include/kamping/v2/collectives/alltoall.hpp
              include/kamping/v2/collectives/alltoallv.hpp
              include/kamping/v2/collectives/barrier.hpp
              include/kamping/v2/collectives/bcast.hpp
              include/kamping/v2/collectives/exscan.hpp
              include/kamping/v2/collectives/gather.hpp
              include/kamping/v2/collectives/gatherv.hpp
              include/kamping/v2/collectives/ibarrier.hpp
              include/kamping/v2/collectives/reduce.hpp
              include/kamping/v2/collectives/scan.hpp
              include/kamping/v2/collectives/scatter.hpp
              include/kamping/v2/collectives/scatterv.hpp
              include/kamping/v2/comm.hpp
              include/kamping/v2/comm_op.hpp
              include/kamping/v2/comm_view_with_pool.hpp
              include/kamping/v2/completion.hpp
              include/kamping/v2/environment.hpp
              include/kamping/v2/error.hpp
              include/kamping/v2/group.hpp
              include/kamping/v2/handle.hpp
              include/kamping/v2/infer.hpp
              include/kamping/v2/info.hpp
              include/kamping/v2/iresult.hpp
              include/kamping/v2/ops.hpp
              include/kamping/v2/p2p.hpp
              include/kamping/v2/p2p/constants.hpp
              include/kamping/v2/p2p/irecv.hpp
              include/kamping/v2/p2p/isend.hpp
              include/kamping/v2/p2p/isendrecv.hpp
              include/kamping/v2/p2p/mrecv.hpp
              include/kamping/v2/p2p/probe.hpp
              include/kamping/v2/p2p/recv.hpp
              include/kamping/v2/p2p/send.hpp
              include/kamping/v2/p2p/send_mode.hpp
              include/kamping/v2/p2p/sendrecv.hpp
              include/kamping/v2/p2p/sendrecv_replace.hpp
              include/kamping/v2/request.hpp
              include/kamping/v2/result.hpp
              include/kamping/v2/sentinels.hpp
              include/kamping/v2/session.hpp
              include/kamping/v2/status.hpp
              include/kamping/v2/type_pool.hpp
              include/kamping/v2/views.hpp
              include/kamping/v2/views/adaptor.hpp
              include/kamping/v2/views/all.hpp
              include/kamping/v2/views/auto_counts_view.hpp
              include/kamping/v2/views/auto_displs_view.hpp
              include/kamping/v2/views/auto_recv.hpp
              include/kamping/v2/views/auto_recv_v.hpp
              include/kamping/v2/views/chunked_view.hpp
              include/kamping/v2/views/concepts.hpp
              include/kamping/v2/views/flatten_v_view.hpp
              include/kamping/v2/views/payload.hpp
              include/kamping/v2/views/ref_single_view.hpp
              include/kamping/v2/views/resize_v_view.hpp
              include/kamping/v2/views/resize_view.hpp
              include/kamping/v2/views/view_interface.hpp
              include/kamping/v2/views/with_count_view.hpp
              include/kamping/v2/views/with_counts_view.hpp
              include/kamping/v2/views/with_displs_view.hpp
              include/kamping/v2/views/with_type_view.hpp
              include/kamping/v2/views/with_value_type_view.hpp
              include/kamping/v2/win.hpp
)
set_target_properties(kamping_v2 PROPERTIES EXPORT_NAME v2)
target_link_libraries(kamping_v2 INTERFACE mpi::core)

if (KAMPING_V2_INSTALL)
    install(
        TARGETS kamping_v2
        EXPORT kamping-v2-v2-targets
        FILE_SET HEADERS
    )
    install(
        EXPORT kamping-v2-v2-targets
        NAMESPACE kamping::
        DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR}
    )
    install(FILES cmake/kamping-v2-v2.cmake DESTINATION ${KAMPING_V2_INSTALL_CMAKEDIR})
endif ()

if (PROJECT_IS_TOP_LEVEL OR KAMPING_V2_BUILD_EXAMPLES_AND_TESTS)
    add_subdirectory(tests)
    add_subdirectory(examples)
endif ()
