// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include "mpi/ops.hpp"

/// @file
/// @brief Re-exports the `make_op` factory and commutativity tags into `kamping::v2`.
///
/// The factory itself and the `handle_traits` glue that make its result flow through the
/// reduce-like collectives live in the core layer (`mpi::experimental`, in `mpi/ops.hpp`);
/// see there for details. This header only surfaces the user-facing names in the ergonomic
/// `kamping::v2` namespace.
///
/// @code
/// // captureless functor on ints, non-commutative by default
/// auto op = kamping::v2::make_op<int>([](int a, int b) { return a + b; });
/// kamping::v2::reduce(data, recv, op, 0, comm);
///
/// // opt in to commutativity for performance
/// auto cop = kamping::v2::make_op<int>(std::plus<int>{}, kamping::v2::commutative);
///
/// // raw MPI callback (no element type needed — the callback is already type-erased)
/// auto rop = kamping::v2::make_op(&my_mpi_callback, kamping::v2::commutative);
/// @endcode

namespace kamping::v2 {

using mpi::experimental::commutative;
using mpi::experimental::commutative_tag;
using mpi::experimental::commutativity_tag;
using mpi::experimental::make_op;
using mpi::experimental::non_commutative;
using mpi::experimental::non_commutative_tag;

} // namespace kamping::v2
