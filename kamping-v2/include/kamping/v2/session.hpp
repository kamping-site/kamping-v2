#pragma once

#include <mpi/session.hpp>

#if defined(MPI_VERSION) && MPI_VERSION >= 4

#include <string_view>

namespace kamping::v2 {

using mpi::experimental::pset_range;
using mpi::experimental::session;
using mpi::experimental::session_accessors;
using mpi::experimental::session_view;

/// @brief Standard MPI-4 process set name constants.
///
/// These are the predefined process sets that every MPI-4 implementation must expose.
namespace psets {

/// @brief Process set containing all processes in the job (`"mpi://WORLD"`).
inline constexpr std::string_view world = "mpi://WORLD";

/// @brief Process set containing only the calling process (`"mpi://SELF"`).
inline constexpr std::string_view self = "mpi://SELF";

} // namespace psets

} // namespace kamping::v2

#endif // MPI_VERSION >= 4
