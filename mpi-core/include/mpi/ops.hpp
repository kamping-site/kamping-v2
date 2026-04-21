#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>

#include <kamping/types/reduce_ops.hpp>
#include <mpi.h>
#include <mpi/handle.hpp>

#include "mpi/buffer.hpp"

/// @file
/// MPI operation contract: customization point (op_traits), accessor dispatch
/// (as_mpi_op), and operation validation concept (valid_op).
///
/// Three-tier accessor dispatch priority:
///   1. op_traits<Op, SBuf, RBuf> specialization — non-intrusive, for types you don't own
///   2. Op convertible to MPI_Op — raw MPI_Op handles pass through
///   3. Builtin op for range element type — delegates to kamping::types::mpi_operation_traits
///
/// To adapt a custom operation non-intrusively, specialize op_traits<Op, SBuf, RBuf>:
///
///   template <>
///   struct op_traits<MyOp, std::vector<int>, std::vector<int>> {
///       static MPI_Op op(MyOp const& op, std::vector<int> const& sbuf, std::vector<int> const& rbuf) {
///           return op.to_mpi_op();
///       }
///   };

namespace mpi::experimental {

// ──────────────────────────────────────────────────────────────────────────────
// Public customization point.
// ──────────────────────────────────────────────────────────────────────────────

/// Trait class for non-intrusive MPI operation customization.
template <typename Op, typename SBuf, typename RBuf>
struct op_traits {};

// ──────────────────────────────────────────────────────────────────────────────
// detail — implementation helpers for the accessor dispatch, not public API.
// ──────────────────────────────────────────────────────────────────────────────

namespace detail {

// Tier-1: op_traits specialization detection
template <typename Op, typename SBuf, typename RBuf>
concept traits_has_op = requires(Op const& op, SBuf const& sbuf, RBuf const& rbuf) {
    { op_traits<std::remove_cvref_t<Op>, SBuf, RBuf>::op(op, sbuf, rbuf) } -> std::convertible_to<MPI_Op>;
};

// Tier-2: builtin MPI_Op handle
template <typename Op>
concept is_builtin_mpi_op = convertible_to_mpi_handle<Op, MPI_Op>;

// Tier-3: range with builtin operation
template <typename Op, typename Buf>
concept is_builtin_range_op =
    std::ranges::range<Buf>
    && kamping::types::mpi_operation_traits<std::remove_cvref_t<Op>, std::ranges::range_value_t<Buf>>::is_builtin;

} // namespace detail

// ──────────────────────────────────────────────────────────────────────────────
// as_mpi_op() — priority: op_traits > builtin handle > range builtin op
// ──────────────────────────────────────────────────────────────────────────────

template <typename Op, send_buffer SBuf, recv_buffer RBuf>
    requires detail::traits_has_op<Op, SBuf, RBuf>
constexpr MPI_Op as_mpi_op(Op const& op, SBuf const& sbuf, RBuf const& rbuf) {
    return op_traits<std::remove_cvref_t<Op>, SBuf, RBuf>::op(op, sbuf, rbuf);
}

template <typename Op, send_buffer SBuf, recv_buffer RBuf>
    requires(!detail::traits_has_op<Op, SBuf, RBuf>) && detail::is_builtin_mpi_op<Op>
constexpr MPI_Op as_mpi_op(Op const& op, SBuf const&, RBuf const&) {
    return handle(op);
}

template <typename Op, send_buffer SBuf, recv_buffer RBuf>
    requires(!detail::traits_has_op<Op, SBuf, RBuf>) && (!detail::is_builtin_mpi_op<Op>)
            && (detail::is_builtin_range_op<Op, SBuf> || detail::is_builtin_range_op<Op, RBuf>)
constexpr MPI_Op as_mpi_op(Op const&, SBuf const&, RBuf const&) {
    using range_buf_t = std::conditional_t<detail::is_builtin_range_op<Op, SBuf>, SBuf, RBuf>;
    return kamping::types::mpi_operation_traits<std::remove_cvref_t<Op>, std::ranges::range_value_t<range_buf_t>>::op();
}

// ──────────────────────────────────────────────────────────────────────────────
// valid_op concept — checks if as_mpi_op is well-defined
// ──────────────────────────────────────────────────────────────────────────────

template <typename Op, typename SBuf, typename RBuf>
concept valid_op = requires(Op const& op, SBuf const& sbuf, RBuf const& rbuf) { as_mpi_op(op, sbuf, rbuf); };

} // namespace mpi::experimental
