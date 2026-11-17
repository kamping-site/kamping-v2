// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <functional>
#include <numeric>
#include <ranges>

#include <mpi.h>

#include "dstl/detail/local_fold.hpp"
#include "dstl/with_subset.hpp"
#include "kamping/v2/buffer.hpp"
#include "kamping/v2/collectives/exscan.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/handle.hpp"
#include "kamping/v2/ops.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "kamping/v2/views/with_type_view.hpp"

namespace dstl {

/// @brief Exclusive prefix scan over the globally distributed range.
///
/// Distributed equivalent of std::exclusive_scan applied to the concatenation of all local
/// ranges across the communicator. The element at global position i in the output is
/// op(init, in[0], ..., in[i-1]).
///
/// R must be a forward_range (two passes are made: one to compute the local total, one
/// for std::exclusive_scan) with a deducible MPI element type. The MPI element type
/// is needed for the scalar exscan across ranks. O must be a contiguous mutable buffer.
///
/// Algorithm:
///   1. Compute local_total via a fold over r (first pass).
///   2. MPI exscan of local totals across non-empty ranks → per-rank prefix.
///   3. std::exclusive_scan over r into out with the rank-adjusted init (second pass).
///
/// Empty ranks are excluded from the exscan via dstl::with_subset; no identity
/// element for op is required.
///
/// @pre out must have the same element count as r on each rank.
/// @pre op must be associative.
template <
    std::ranges::forward_range                       R,
    kamping::v2::recv_buffer                         O,
    kamping::v2::convertible_to_mpi_handle<MPI_Comm> Comm = kamping::v2::comm_view,
    class BinaryOp                                        = std::plus<>>
    requires kamping::v2::has_mpi_type<R> && kamping::v2::reduction_op<BinaryOp, std::ranges::range_value_t<R>>
void exclusive_scan(
    R&&                           r,
    O&&                           out,
    std::ranges::range_value_t<R> init,
    BinaryOp const&               op   = {},
    Comm const&                   comm = kamping::v2::comm_world()
) {
    using T                               = std::ranges::range_value_t<R>;
    MPI_Datatype const           mpi_type = kamping::v2::type(r);
    kamping::v2::comm_view const cv{kamping::v2::handle(comm)};
    // Applies op locally: op itself if it is callable, otherwise its MPI_Op through MPI_Reduce_local.
    auto const local_op = kamping::v2::as_local_op<T>(op, mpi_type);

    T rank_prefix{};
    with_subset(cv, !std::ranges::empty(r), [&](kamping::v2::comm_view sub_comm) {
        // First pass: compute local total without assuming T{} is the identity for op. op need only be associative,
        // so the elements are reordered only if op is commutative.
        T const local_total = detail::local_fold<T>(std::ranges::begin(r), std::ranges::end(r), op, mpi_type);

        if (sub_comm.size() > 1) {
            kamping::v2::exscan(
                kamping::v2::views::ref_single(local_total) | kamping::v2::views::with_type(mpi_type),
                kamping::v2::views::ref_single(rank_prefix) | kamping::v2::views::with_type(mpi_type),
                op,
                sub_comm
            );
        }

        T const global_init = (sub_comm.rank() == 0) ? init : local_op(init, rank_prefix);
        // Second pass: local exclusive scan with the rank-adjusted initial value.
        std::exclusive_scan(std::ranges::begin(r), std::ranges::end(r), std::ranges::begin(out), global_init, local_op);
    });
}

} // namespace dstl
