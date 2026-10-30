// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <mpi/win.hpp>

namespace kamping::v2 {
using mpi::experimental::fence;
using mpi::experimental::win;
using mpi::experimental::win_accessors;
using mpi::experimental::win_allocate;
using mpi::experimental::win_view;
} // namespace kamping::v2
