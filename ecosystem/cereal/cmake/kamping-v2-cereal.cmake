# The kamping-v2 component 'cereal': kamping::v2::cereal.
include_guard()
include("${CMAKE_CURRENT_LIST_DIR}/kamping-v2-v2.cmake")
find_dependency(cereal 1.3.2)
include("${CMAKE_CURRENT_LIST_DIR}/kamping-v2-cereal-targets.cmake")
