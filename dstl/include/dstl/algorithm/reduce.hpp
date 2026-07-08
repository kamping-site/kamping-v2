// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <functional>
#include <numeric>
#include <optional>
#include <ranges>

#include <mpi.h>

#include "kamping/v2/collectives/allreduce.hpp"
#include "kamping/v2/collectives/reduce.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/sentinels.hpp"
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
/// for op when empty ranks are possible.
///
/// @pre op must be associative, commutative, and mappable to an MPI_Op.
/// @pre If any rank's range is empty, T{} must be the identity element for op.
template <
    std::ranges::forward_range                             R,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm,
    class BinaryOp                                              = std::plus<>>
    requires mpi::experimental::has_mpi_type<R>
auto reduce(R&& r, std::ranges::range_value_t<R> init = {}, BinaryOp op = {}, Comm const& comm = MPI_COMM_WORLD)
    -> std::ranges::range_value_t<R> {
    using T                     = std::ranges::range_value_t<R>;
    MPI_Datatype const mpi_type = mpi::experimental::type(r);

    auto first = std::ranges::begin(r);
    auto last  = std::ranges::end(r);
    T    result = (first == last) ? T{} : std::reduce(std::next(first), last, *first, op);

    kamping::v2::allreduce(
        kamping::v2::inplace,
        kamping::v2::views::ref_single(result) | kamping::v2::views::with_type(mpi_type),
        op,
        comm
    );

    return op(init, result);
}

/// @brief Reduces the globally distributed range to a scalar on the root rank.
///
/// Distributed equivalent of std::reduce applied to the concatenation of all local ranges,
/// but the result is returned only on the root rank. Non-root ranks receive std::nullopt.
/// Returns std::optional containing op(init, in[0], ..., in[N-1]) on root, and
/// std::nullopt on all other ranks.
///
/// @pre op must be associative, commutative, and mappable to an MPI_Op.
/// @pre If any rank's range is empty, T{} must be the identity element for op.
template <
    std::ranges::forward_range                             R,
    mpi::experimental::rank                                Rank = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm,
    class BinaryOp                                              = std::plus<>>
    requires mpi::experimental::has_mpi_type<R>
auto reduce_to_root(
    R&& r, std::ranges::range_value_t<R> init = {}, BinaryOp op = {}, Rank root = {}, Comm const& comm = MPI_COMM_WORLD
) -> std::optional<std::ranges::range_value_t<R>> {
    using T                     = std::ranges::range_value_t<R>;
    MPI_Datatype const mpi_type = mpi::experimental::type(r);

    auto first  = std::ranges::begin(r);
    auto last   = std::ranges::end(r);
    T    result = (first == last) ? T{} : std::reduce(std::next(first), last, *first, op);

    kamping::v2::comm_view cv{mpi::experimental::handle(comm)};

    if (cv.rank() == mpi::experimental::to_rank(root)) {
        kamping::v2::reduce(
            kamping::v2::inplace,
            kamping::v2::views::ref_single(result) | kamping::v2::views::with_type(mpi_type),
            op,
            root,
            comm
        );
        return op(init, result);
    } else {
        kamping::v2::reduce(
            kamping::v2::views::ref_single(result) | kamping::v2::views::with_type(mpi_type),
            op,
            root,
            comm
        );
        return std::nullopt;
    }
}

} // namespace dstl
