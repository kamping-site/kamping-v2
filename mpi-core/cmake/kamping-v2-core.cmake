# The kamping-v2 component 'core': mpi::core.
include_guard()
include("${CMAKE_CURRENT_LIST_DIR}/kamping-v2-types.cmake")
find_dependency(MPI COMPONENTS CXX)
find_dependency(kamping_v2 CONFIG PATHS "${CMAKE_CURRENT_LIST_DIR}/../kamping/v2" NO_DEFAULT_PATH)
include("${CMAKE_CURRENT_LIST_DIR}/kamping-v2-core-targets.cmake")
