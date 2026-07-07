// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include "kamping/v2/comm_view_with_pool.hpp" // IWYU pragma: export
#include <mpi/comm.hpp>

namespace kamping::v2 {
using mpi::experimental::comm;
using mpi::experimental::comm_accessors;
using mpi::experimental::comm_view;
using mpi::experimental::dup;
using mpi::experimental::split;
} // namespace kamping::v2
