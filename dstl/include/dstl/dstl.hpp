// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

/// @file
/// Umbrella header for the dstl collectives layer: grid and flat communicators, factoring strategies,
/// execution/ordering tags, the grid and flat alltoallv wrappers, and visualization helpers.

#include "dstl/alltoallv.hpp"          // IWYU pragma: export
#include "dstl/factoring.hpp"           // IWYU pragma: export
#include "dstl/thread_multiple_comm.hpp" // IWYU pragma: export
#include "dstl/grid_comm.hpp"           // IWYU pragma: export
#include "dstl/tags.hpp"                // IWYU pragma: export
#include "dstl/visualization.hpp"       // IWYU pragma: export
#include "dstl/algorithm.hpp"           // IWYU pragma: export
#include "dstl/with_subset.hpp"         // IWYU pragma: export
