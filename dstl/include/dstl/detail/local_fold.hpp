// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include "kamping/v2/buffer.hpp"
#include "kamping/v2/ops.hpp"

namespace dstl::detail {

// Op is applied locally through MPI_Reduce_local: it is not callable (e.g. a raw MPI_Op), or callable only through it,
// as op and op_view are.
template <typename Op, typename T>
inline constexpr bool applied_through_mpi =
    !std::is_invocable_v<Op const&, T const&, T const&>
    || std::derived_from<std::remove_cvref_t<Op>, kamping::v2::op_accessors<std::remove_cvref_t<Op>>>;

/// Folds the non-empty range [first, last) with op on this process: in[0] op in[1] op ... op in[n-1].
///
/// op need only be associative; the elements are reordered only where op is known to be commutative:
///   - commutative at compile time (kamping::v2::commutative_op, e.g. std::plus<> or a commutative functor_op):
///     std::reduce with the functor, which may reassociate and vectorize;
///   - applied through MPI (a raw MPI_Op, op, op_view), commutative at runtime, on a contiguous range:
///     halving, with log2(n) calls of MPI_Reduce_local on a copy, each over many elements, so that the loops of the MPI
///     library do the work instead of one MPI call per element;
///   - otherwise: std::accumulate, in order, with as_local_op (for an MPI operation, one MPI_Reduce_local per element).
///
/// dt is the datatype of one element; it describes a T with an extent of sizeof(T).
template <typename T, std::forward_iterator It, typename Op>
    requires kamping::v2::reduction_op<Op, T>
T local_fold(It first, It last, Op const& op, MPI_Datatype dt) {
    auto const local_op = kamping::v2::as_local_op<T>(op, dt);
    T          head     = *first;
    ++first;
    if constexpr (kamping::v2::commutative_op<Op, T>) {
        return std::reduce(first, last, std::move(head), local_op);
    } else {
        if constexpr (std::contiguous_iterator<It> && applied_through_mpi<Op, T>) {
            if (kamping::v2::op_commutative(op)) {
                std::vector<T> buf;
                buf.reserve(static_cast<std::size_t>(std::ranges::distance(first, last)) + 1);
                buf.push_back(std::move(head));
                buf.insert(buf.end(), first, last);
                // Combine the upper half into the lower half until one element is left; with a commutative op, the
                // order of the elements does not matter.
                std::size_t n = buf.size();
                while (n > 1) {
                    std::size_t const half = n / 2;
                    kamping::v2::reduce_local(
                        kamping::v2::mpi_cspan(buf.data() + (n - half), static_cast<std::ptrdiff_t>(half), dt),
                        kamping::v2::mpi_span(buf.data(), static_cast<std::ptrdiff_t>(half), dt),
                        op
                    );
                    n -= half;
                }
                return std::move(buf.front());
            }
        }
        return std::accumulate(first, last, std::move(head), local_op);
    }
}

} // namespace dstl::detail
