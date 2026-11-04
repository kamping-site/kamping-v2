# The kamping-v2 component 'thrust': kamping::v2::thrust.
include_guard()
include("${CMAKE_CURRENT_LIST_DIR}/kamping-v2-v2.cmake")
find_dependency(CUDAToolkit)
include("${CMAKE_CURRENT_LIST_DIR}/kamping-v2-thrust-targets.cmake")
