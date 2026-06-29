// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <functional>
#include <numeric>
#include <ranges>

#include <mpi.h>

#include "kamping/v2/collectives/allreduce.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "kamping/v2/views/with_type_view.hpp"
#include "mpi/buffer.hpp"
#include "mpi/handle.hpp"

namespace dstl {

/// @brief Reduces the globally distributed range to a scalar, available on all ranks.
///
/// Distributed equivalent of std::reduce applied to the concatenation of all local ranges.
/// Returns op(init, in[0], in[1], ..., in[N-1]) where N is the global element count.
///
/// R must be a forward_range with a deducible MPI element type (contiguous layout is not
/// required; only the element type determines the MPI_Datatype for the scalar allreduce).
/// Each rank computes its local reduction then participates in an allreduce.
/// Empty ranks contribute T{} to the allreduce, so T{} must be the identity element
/// for op (true for std::plus<> / std::multiplies<> with their respective identity values).
///
/// @pre op must be associative, commutative, and mappable to an MPI_Op.
/// @pre T{} must be the identity element for op on empty ranks.
template <
    std::ranges::forward_range                              R,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm     = MPI_Comm,
    class                                                   BinaryOp = std::plus<>>
    requires mpi::experimental::has_mpi_type<R>
auto reduce(
    R&&                           r,
    std::ranges::range_value_t<R> init = {},
    BinaryOp                      op   = {},
    Comm const&                   comm = MPI_COMM_WORLD
) -> std::ranges::range_value_t<R> {
    using T                    = std::ranges::range_value_t<R>;
    MPI_Datatype const mpi_type = mpi::experimental::type(r);

    // Local reduction: T{} serves as the identity for empty ranks.
    T local_result = std::reduce(std::ranges::begin(r), std::ranges::end(r), T{}, op);

    T global_result{};
    kamping::v2::allreduce(
        kamping::v2::views::ref_single(local_result) | kamping::v2::views::with_type(mpi_type),
        kamping::v2::views::ref_single(global_result) | kamping::v2::views::with_type(mpi_type),
        op,
        comm
    );

    return op(init, global_result);
}

} // namespace dstl
