// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

#include <kamping/types/reduce_ops.hpp>
#include <mpi.h>

#include "mpi/handle.hpp"

/// @file
/// @brief `kamping::v2::make_op` — ergonomic factory for user-defined MPI reduction operations.
///
/// `make_op` is a thin factory over the RAII operation handles from the `kamping-types` dependency.
/// It picks the right one for the argument it is given and performs the `MPI_Op_create` under the
/// hood; the actual reduction logic lives in `kamping-types`:
///
///   - a raw MPI callback (`void(*)(void*, void*, int*, MPI_Datatype*)`) → `kamping::types::ScopedCallbackOp`
///   - a default-constructible binary functor / captureless lambda      → `kamping::types::ScopedFunctorOp`
///
/// The returned handle works directly with the reduce-like `kamping::v2::` collectives
/// (`reduce`, `allreduce`, `scan`, `exscan`, …): the `handle_traits` specializations at the bottom
/// of this file teach `mpi::experimental::as_mpi_op` to extract the underlying `MPI_Op`, so the
/// handle satisfies the `valid_op` concept out of the box.
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
/// @brief Maps a commutativity tag type to the `bool` template argument expected by the scoped ops.
template <commutativity_tag C>
inline constexpr bool is_commutative_v = std::is_same_v<C, commutative_tag>;

/// @brief The raw MPI reduction callback signature.
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
/// @param commutativity `kamping::v2::commutative` or `kamping::v2::non_commutative`.
/// @return A `kamping::types::ScopedCallbackOp` owning the freshly created `MPI_Op`.
template <commutativity_tag Commutativity = non_commutative_tag>
[[nodiscard]] auto make_op(detail::op_callback_t callback, Commutativity commutativity = {}) {
    static_cast<void>(commutativity);
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
/// @param commutativity `kamping::v2::commutative` or `kamping::v2::non_commutative`.
/// @return A `kamping::types::ScopedFunctorOp` owning the freshly created `MPI_Op`.
template <typename T, typename Op, commutativity_tag Commutativity = non_commutative_tag>
    requires std::is_default_constructible_v<std::remove_cvref_t<Op>>
             && std::is_invocable_r_v<T, std::remove_cvref_t<Op>, T const&, T const&>
[[nodiscard]] auto make_op(Op&& op, Commutativity commutativity = {}) {
    static_cast<void>(commutativity);
    return kamping::types::ScopedFunctorOp<detail::is_commutative_v<Commutativity>, T, std::remove_cvref_t<Op>>{
        std::forward<Op>(op)
    };
}

} // namespace kamping::v2

namespace mpi::experimental {

// ──────────────────────────────────────────────────────────────────────────────
// handle_traits specializations — let as_mpi_op() extract the MPI_Op from a scoped op.
//
// These make any kamping::types::ScopedFunctorOp / ScopedCallbackOp (however it was created)
// satisfy convertible_to_mpi_handle<MPI_Op>, so as_mpi_op() picks it up via its builtin-handle
// tier and the op flows straight into the reduce-like collectives.
// ──────────────────────────────────────────────────────────────────────────────

/// @brief Extracts the underlying `MPI_Op` from a `ScopedFunctorOp`.
template <bool Commutative, typename T, typename Op>
struct handle_traits<kamping::types::ScopedFunctorOp<Commutative, T, Op>> {
    /// @returns the wrapped `MPI_Op`.
    static MPI_Op handle(kamping::types::ScopedFunctorOp<Commutative, T, Op> const& op) noexcept {
        return op.get();
    }
};

/// @brief Extracts the underlying `MPI_Op` from a `ScopedCallbackOp`.
template <bool Commutative>
struct handle_traits<kamping::types::ScopedCallbackOp<Commutative>> {
    /// @returns the wrapped `MPI_Op`.
    static MPI_Op handle(kamping::types::ScopedCallbackOp<Commutative> const& op) noexcept {
        return op.get();
    }
};

} // namespace mpi::experimental
