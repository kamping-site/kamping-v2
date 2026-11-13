// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

/// @file
/// Rank types with neighbor arithmetic in the `kamping::v2` namespace, re-exported from the core layer
/// (`mpi/rank.hpp`): `periodic_rank` wraps around at the ends of the communicator, `nonperiodic_rank` stands for
/// `MPI_PROC_NULL` beyond them.

#pragma once

#include "mpi/rank.hpp"

namespace kamping::v2 {
using mpi::experimental::nonperiodic_rank;
using mpi::experimental::periodic_rank;
} // namespace kamping::v2
