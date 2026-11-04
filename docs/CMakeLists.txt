# Documentation: Doxygen extracts the API from the headers as XML, Sphinx renders the site from docs/*.md and that XML.
# Build with `cmake --build <build-dir> --target docs`; the site ends up in <build-dir>/docs/html.

find_package(Doxygen 1.10 REQUIRED)
find_program(SPHINX_BUILD sphinx-build REQUIRED DOC "sphinx-build from docs/requirements.txt")

set(DOXYGEN_GENERATE_HTML NO)
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_XML_PROGRAMLISTING NO)
set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_FILE_PATTERNS *.hpp)
set(DOXYGEN_EXCLUDE_SYMBOLS *::detail::* *::detail)
set(DOXYGEN_PREDEFINED MPI_VERSION=4)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_EXTRACT_STATIC YES)
set(DOXYGEN_INLINE_INHERITED_MEMB YES) # e.g. rank() of comm_view, from its accessor base
set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
set(DOXYGEN_WARN_LOGFILE ${CMAKE_CURRENT_BINARY_DIR}/doxygen/warnings.log)
set(DOXYGEN_QUIET YES)
# show includes as users write them: <mpi/buffer.hpp>, <kamping/v2/...>
set(DOXYGEN_STRIP_FROM_PATH ${PROJECT_SOURCE_DIR})
set(DOXYGEN_STRIP_FROM_INC_PATH
    ${PROJECT_SOURCE_DIR}/mpi-core/include
    ${PROJECT_SOURCE_DIR}/kamping-v2/include
    ${PROJECT_SOURCE_DIR}/dstl/include
    ${PROJECT_SOURCE_DIR}/ecosystem/cereal/include
    ${PROJECT_SOURCE_DIR}/ecosystem/kokkos/include
    ${PROJECT_SOURCE_DIR}/ecosystem/sycl/include
    ${PROJECT_SOURCE_DIR}/ecosystem/thrust/include
)

doxygen_add_docs(
    kamping_v2_doxygen
    ${PROJECT_SOURCE_DIR}/mpi-core/include
    ${PROJECT_SOURCE_DIR}/kamping-v2/include
    ${PROJECT_SOURCE_DIR}/dstl/include
    ${PROJECT_SOURCE_DIR}/ecosystem/cereal/include
    ${PROJECT_SOURCE_DIR}/ecosystem/kokkos/include
    ${PROJECT_SOURCE_DIR}/ecosystem/sycl/include
    ${PROJECT_SOURCE_DIR}/ecosystem/thrust/include
    COMMENT "Extracting the API with Doxygen"
)

set(_kamping_v2_sphinx_options -n -q)
if (KAMPING_V2_WARNINGS_ARE_ERRORS)
    list(APPEND _kamping_v2_sphinx_options -W --keep-going)
endif ()

add_custom_target(
    docs
    COMMAND
        ${CMAKE_COMMAND} -E env KAMPING_V2_DOXYGEN_XML=${CMAKE_CURRENT_BINARY_DIR}/doxygen/xml ${SPHINX_BUILD} -b html
        ${_kamping_v2_sphinx_options} -d ${CMAKE_CURRENT_BINARY_DIR}/doctrees ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/html
    COMMENT "Building the documentation with Sphinx"
    VERBATIM
)
add_dependencies(docs kamping_v2_doxygen)
