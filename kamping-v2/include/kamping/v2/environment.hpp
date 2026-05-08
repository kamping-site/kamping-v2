// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

/// @file
/// @brief Re-exports \c mpi::experimental::environment into the \c kamping::v2 namespace.

#pragma once

#include "mpi/environment.hpp"

namespace kamping::v2 {

using mpi::experimental::environment;

} // namespace kamping::v2
