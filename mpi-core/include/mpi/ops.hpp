// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>
#include <utility>

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
// handle_traits specializations — let as_mpi_op() extract the MPI_Op from a scoped op.
//
// These make any kamping::types::ScopedFunctorOp / ScopedCallbackOp (however it was created)
// satisfy convertible_to_mpi_handle<MPI_Op>, so as_mpi_op() picks it up via its builtin-handle
// tier and the op flows straight into the reduce-like collectives.
// ──────────────────────────────────────────────────────────────────────────────

/// Extracts the underlying MPI_Op from a ScopedFunctorOp.
template <bool Commutative, typename T, typename Op>
struct handle_traits<kamping::types::ScopedFunctorOp<Commutative, T, Op>> {
    /// @returns the wrapped MPI_Op.
    static MPI_Op handle(kamping::types::ScopedFunctorOp<Commutative, T, Op> const& op) noexcept {
        return op.get();
    }
};

/// Extracts the underlying MPI_Op from a ScopedCallbackOp.
template <bool Commutative>
struct handle_traits<kamping::types::ScopedCallbackOp<Commutative>> {
    /// @returns the wrapped MPI_Op.
    static MPI_Op handle(kamping::types::ScopedCallbackOp<Commutative> const& op) noexcept {
        return op.get();
    }
};

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

// ──────────────────────────────────────────────────────────────────────────────
// make_op — ergonomic factory for user-defined MPI reduction operations
//
// A thin factory over the RAII operation handles from the kamping-types dependency. It picks the
// right one for the argument it is given and performs the MPI_Op_create under the hood; the actual
// reduction logic lives in kamping-types:
//
//   - a raw MPI callback (void(*)(void*, void*, int*, MPI_Datatype*)) → ScopedCallbackOp
//   - a default-constructible binary functor / captureless lambda     → ScopedFunctorOp
//
// The returned handle works directly with the reduce-like collectives via the handle_traits
// specializations above, which teach as_mpi_op() to extract the underlying MPI_Op.
// ──────────────────────────────────────────────────────────────────────────────

/// @brief Commutativity tag type: the operation is commutative (re-exported from `kamping::ops`).
using commutative_tag = kamping::ops::internal::commutative_tag;
/// @brief Commutativity tag type: the operation is non-commutative (re-exported from `kamping::ops`).
using non_commutative_tag = kamping::ops::internal::non_commutative_tag;

/// @brief Tag value marking an operation as commutative (re-exported from `kamping::ops`).
inline constexpr commutative_tag commutative{};
/// @brief Tag value marking an operation as non-commutative (re-exported from `kamping::ops`).
inline constexpr non_commutative_tag non_commutative{};

/// @brief Concept satisfied by the two commutativity tag types.
template <typename C>
concept commutativity_tag = std::same_as<C, commutative_tag> || std::same_as<C, non_commutative_tag>;

namespace detail {
/// Maps a commutativity tag type to the `bool` template argument expected by the scoped ops.
template <commutativity_tag C>
inline constexpr bool is_commutative_v = std::is_same_v<C, commutative_tag>;

/// The raw MPI reduction callback signature.
using op_callback_t = void (*)(void*, void*, int*, MPI_Datatype*);
} // namespace detail

/// @brief Creates an `MPI_Op` from a raw MPI reduction callback.
///
/// Use this when you already have a function with MPI's callback signature
/// `void(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype)`. No element type is
/// required because the callback is already type-erased.
///
/// @tparam Commutativity Commutativity tag type; defaults to non-commutative.
/// @param callback Non-null MPI callback function pointer.
/// @return A `kamping::types::ScopedCallbackOp` owning the freshly created `MPI_Op`.
template <commutativity_tag Commutativity = non_commutative_tag>
[[nodiscard]] auto make_op(detail::op_callback_t callback, Commutativity = {}) {
    return kamping::types::ScopedCallbackOp<detail::is_commutative_v<Commutativity>>{callback};
}

/// @brief Creates an `MPI_Op` from a default-constructible binary functor.
///
/// Handles stateless functors and captureless lambdas (which are default-constructible in C++20).
/// The element-wise application is provided by `kamping::types::ScopedFunctorOp`; the functor must
/// be callable as `T(T const&, T const&)`.
///
/// Stateful (capturing) lambdas are intentionally not supported here — they are neither
/// default-constructible nor convertible to a raw callback. Wrap the required state in a
/// default-constructible functor, or create the callback yourself and pass it to the overload above.
///
/// @tparam T The element type the operation is applied to. Must be specified explicitly, e.g.
///           `make_op<int>(...)`.
/// @tparam Op Functor type (deduced).
/// @tparam Commutativity Commutativity tag type; defaults to non-commutative.
/// @param op The binary operation.
/// @return A `kamping::types::ScopedFunctorOp` owning the freshly created `MPI_Op`.
template <typename T, typename Op, commutativity_tag Commutativity = non_commutative_tag>
    requires std::is_default_constructible_v<std::remove_cvref_t<Op>>
             && std::is_invocable_r_v<T, std::remove_cvref_t<Op>, T const&, T const&>
[[nodiscard]] auto make_op(Op&& op, Commutativity = {}) {
    return kamping::types::ScopedFunctorOp<detail::is_commutative_v<Commutativity>, T, std::remove_cvref_t<Op>>{
        std::forward<Op>(op)
    };
}

} // namespace mpi::experimental
