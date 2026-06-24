// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

/// @file
/// Umbrella header for the dstl all-to-all-v wrappers. Pulls in both the grid (message-combining)
/// variant `dstl::grid_alltoallv` dispatched on `dstl::grid_comm` and the flat (single-exchange)
/// variant `dstl::alltoallv` dispatched on `dstl::thread_multiple_comm`.

#include "dstl/flat_alltoallv.hpp"
#include "dstl/grid_alltoallv.hpp"
