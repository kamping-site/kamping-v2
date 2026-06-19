// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

/// @file
/// Umbrella header for the dstl all-to-all-v wrappers. Pulls in both the grid (message-combining)
/// variant dispatched on `dstl::grid_comm` and the flat (single-exchange) variant dispatched on
/// `dstl::thread_multiple_comm`. Include this to get every `dstl::alltoallv` overload.

#include "dstl/flat_alltoallv.hpp"
#include "dstl/grid_alltoallv.hpp"
