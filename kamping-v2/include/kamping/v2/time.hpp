// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include "mpi/time.hpp"

/// @file
/// `clock`, a `std::chrono` clock reading `MPI_Wtime`, re-exported from the core layer (`mpi/time.hpp`).

namespace kamping::v2 {
using mpi::experimental::clock;
} // namespace kamping::v2
