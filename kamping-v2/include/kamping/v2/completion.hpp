// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include "mpi/completion/wait.hpp"
#include "mpi/completion/waitall.hpp"
#include "mpi/completion/waitany.hpp"
#include "mpi/completion/waitsome.hpp"

namespace kamping::v2 {
using mpi::experimental::test;
using mpi::experimental::testall;
using mpi::experimental::testany;
using mpi::experimental::testany_result;
using mpi::experimental::testsome;
using mpi::experimental::wait;
using mpi::experimental::waitall;
using mpi::experimental::waitany;
using mpi::experimental::waitsome;
} // namespace kamping::v2
