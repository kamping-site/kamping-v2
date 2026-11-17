// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include "mpi/op.hpp"
#include "mpi/ops.hpp"

/// @file
/// @brief Reduction operations in `kamping::v2`, re-exported from the core layer (`mpi/op.hpp`, `mpi/ops.hpp`): the
/// functors of the predefined operations (`ops::max`, ...), the operation objects (`op` and `functor_op`
/// from `make_op`, the non-owning `op_view`), `reduce_local` and the commutativity tags.
///
/// @code
/// // predefined operation: std::plus<> on int is MPI_SUM
/// kamping::v2::allreduce(data, sums, std::plus<>{}, comm);
/// kamping::v2::allreduce(data, maxima, kamping::v2::ops::max<>{}, comm);
///
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
namespace ops {
using mpi::experimental::ops::bit_and;
using mpi::experimental::ops::bit_or;
using mpi::experimental::ops::bit_xor;
using mpi::experimental::ops::logical_and;
using mpi::experimental::ops::logical_or;
using mpi::experimental::ops::logical_xor;
using mpi::experimental::ops::max;
using mpi::experimental::ops::min;
using mpi::experimental::ops::multiplies;
using mpi::experimental::ops::plus;
} // namespace ops

using mpi::experimental::as_local_op;
using mpi::experimental::builtin_op;
using mpi::experimental::builtin_op_handle;
using mpi::experimental::commutative;
using mpi::experimental::commutative_tag;
using mpi::experimental::commutativity_tag;
using mpi::experimental::functor_op;
using mpi::experimental::make_op;
using mpi::experimental::non_commutative;
using mpi::experimental::non_commutative_tag;
using mpi::experimental::op;
using mpi::experimental::op_accessors;
using mpi::experimental::op_view;
using mpi::experimental::reduce_local;
using mpi::experimental::reduction_functor;
using mpi::experimental::reduction_op;
using mpi::experimental::user_function;
using mpi::experimental::valid_op;
} // namespace kamping::v2
