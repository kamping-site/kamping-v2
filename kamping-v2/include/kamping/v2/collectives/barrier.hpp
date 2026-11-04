// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi.h>

#include "mpi/collectives/barrier.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {
using mpi::experimental::barrier;
} // namespace kamping::v2
