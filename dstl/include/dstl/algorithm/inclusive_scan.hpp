// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <functional>
#include <numeric>
#include <ranges>

#include <mpi.h>

#include "dstl/with_subset.hpp"
#include "kamping/v2/collectives/exscan.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "kamping/v2/views/with_type_view.hpp"
#include "mpi/buffer.hpp"
#include "mpi/handle.hpp"

namespace dstl {

/// @brief Inclusive prefix scan over the globally distributed range.
///
/// Distributed equivalent of std::inclusive_scan applied to the concatenation of all local
/// ranges across the communicator. The element at global position i in the output is
/// op(in[0], ..., in[i]).
///
/// R must be a forward_range (for consistency; only one pass is made) with a deducible
/// MPI element type. O must be a contiguous mutable buffer (recv_buffer).
///
/// Algorithm:
///   1. Local std::inclusive_scan; the last element of out is the local total.
///   2. MPI exscan of local totals across non-empty ranks → per-rank prefix.
///   3. For ranks with a predecessor: apply op(rank_prefix, out[i]) to every local output.
///
/// Empty ranks are excluded from the exscan via dstl::with_subset, so no identity
/// element for op is required.
///
/// @pre out must have the same element count as r on each rank.
/// @pre op must be associative and mappable to an MPI_Op.
template <
    std::ranges::forward_range                              R,
    mpi::experimental::recv_buffer                          O,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm     = MPI_Comm,
    class                                                   BinaryOp = std::plus<>>
    requires mpi::experimental::has_mpi_type<R>
void inclusive_scan(
    R&&         r,
    O&&         out,
    BinaryOp    op   = {},
    Comm const& comm = MPI_COMM_WORLD
) {
    using T                      = std::ranges::range_value_t<R>;
    MPI_Datatype const mpi_type  = mpi::experimental::type(r);
    kamping::v2::comm_view const cv{mpi::experimental::handle(comm)};

    T rank_prefix{};
    with_subset(cv, !std::ranges::empty(r), [&](kamping::v2::comm_view sub_comm) {
        std::inclusive_scan(std::ranges::begin(r), std::ranges::end(r), std::ranges::begin(out), op);
        T const local_total = *std::prev(std::ranges::end(out));

        if (sub_comm.size() > 1) {
            kamping::v2::exscan(
                kamping::v2::views::ref_single(local_total) | kamping::v2::views::with_type(mpi_type),
                kamping::v2::views::ref_single(rank_prefix) | kamping::v2::views::with_type(mpi_type),
                op,
                sub_comm
            );
        }

        // Rank 0 of the sub-communicator has no predecessor; its local scan is already correct.
        // All other ranks shift their output up by the cumulative prefix from previous ranks.
        if (sub_comm.rank() > 0) {
            for (auto& x : out) {
                x = op(rank_prefix, x);
            }
        }
    });
}

} // namespace dstl
