// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <ranges>
#include <type_traits>

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/builtin_types.hpp"
#include "mpi/handle.hpp"

/// @file
/// MPI operation contract: the functors for predefined operations (`ops::max`, ...), their mapping to predefined
/// `MPI_Op`s (`builtin_op`), the customization point `op_traits`, the accessor dispatch `as_mpi_op` and the concept
/// `valid_op`.
///
/// Three-tier accessor dispatch priority:
///   1. op_traits<Op, SBuf, RBuf> specialization — non-intrusive, for types you don't own
///   2. Op convertible to MPI_Op — raw MPI_Op handles and operation objects (`op`, `op_view`, ...) pass through
///   3. Functor of a predefined operation on the range element type — `builtin_op<Op, T>`
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

namespace ops {

/// @brief Function object returning the larger of two values, mapped to `MPI_MAX`.
///
/// The standard library has no function object for `std::max`. `max<>` deduces the operand type.
template <typename T = void>
struct max {
    /// @return The larger of `lhs` and `rhs` (`std::max`).
    constexpr T operator()(T const& lhs, T const& rhs) const {
        return std::max(lhs, rhs);
    }
};

/// @brief Function object returning the larger of two values of a deduced type, mapped to `MPI_MAX`.
template <>
struct max<void> {
    /// @return The larger of `lhs` and `rhs` (`std::max`).
    template <typename T>
    constexpr T operator()(T const& lhs, T const& rhs) const {
        return std::max(lhs, rhs);
    }
};

/// @brief Function object returning the smaller of two values, mapped to `MPI_MIN`.
///
/// The standard library has no function object for `std::min`. `min<>` deduces the operand type.
template <typename T = void>
struct min {
    /// @return The smaller of `lhs` and `rhs` (`std::min`).
    constexpr T operator()(T const& lhs, T const& rhs) const {
        return std::min(lhs, rhs);
    }
};

/// @brief Function object returning the smaller of two values of a deduced type, mapped to `MPI_MIN`.
template <>
struct min<void> {
    /// @return The smaller of `lhs` and `rhs` (`std::min`).
    template <typename T>
    constexpr T operator()(T const& lhs, T const& rhs) const {
        return std::min(lhs, rhs);
    }
};

/// @brief Function object for the logical exclusive or, mapped to `MPI_LXOR`.
///
/// The standard library has no function object for it. `logical_xor<>` deduces the operand types.
template <typename T = void>
struct logical_xor {
    /// @return Whether exactly one of `lhs` and `rhs` is true.
    constexpr bool operator()(T const& lhs, T const& rhs) const {
        return static_cast<bool>(lhs) != static_cast<bool>(rhs);
    }
};

/// @brief Function object for the logical exclusive or of deduced operand types, mapped to `MPI_LXOR`.
template <>
struct logical_xor<void> {
    /// @return Whether exactly one of `lhs` and `rhs` is true.
    template <typename T, typename U>
    constexpr bool operator()(T const& lhs, U const& rhs) const {
        return static_cast<bool>(lhs) != static_cast<bool>(rhs);
    }
};

/// @brief `std::plus`, mapped to `MPI_SUM`.
template <typename T = void>
using plus = std::plus<T>;
/// @brief `std::multiplies`, mapped to `MPI_PROD`.
template <typename T = void>
using multiplies = std::multiplies<T>;
/// @brief `std::logical_and`, mapped to `MPI_LAND`.
template <typename T = void>
using logical_and = std::logical_and<T>;
/// @brief `std::logical_or`, mapped to `MPI_LOR`.
template <typename T = void>
using logical_or = std::logical_or<T>;
/// @brief `std::bit_and`, mapped to `MPI_BAND`.
template <typename T = void>
using bit_and = std::bit_and<T>;
/// @brief `std::bit_or`, mapped to `MPI_BOR`.
template <typename T = void>
using bit_or = std::bit_or<T>;
/// @brief `std::bit_xor`, mapped to `MPI_BXOR`.
template <typename T = void>
using bit_xor = std::bit_xor<T>;

} // namespace ops

// ──────────────────────────────────────────────────────────────────────────────
// Predefined operations
// ──────────────────────────────────────────────────────────────────────────────

namespace detail {

// Op is F<void> or F<T>.
template <template <typename> class F, typename Op, typename T>
concept functor_of = std::same_as<Op, F<void>> || std::same_as<Op, F<T>>;

// T is an arithmetic builtin type in one of the categories Cs.
template <typename T, builtin_category... Cs>
concept builtin_in_category = arithmetic_builtin<T> && ((builtin_type_info<T>::category == Cs) || ...);

enum class predefined_op { none, max, min, sum, prod, land, lor, lxor, band, bor, bxor };

// The predefined operation that applies the functor Op to elements of T, following the table of MPI 4.1, Section
// 6.9.2.
template <typename Op, typename T>
consteval predefined_op predefined_op_of() {
    using enum builtin_category;
    if constexpr (functor_of<ops::max, Op, T> && builtin_in_category<T, integer, floating>) {
        return predefined_op::max;
    } else if constexpr (functor_of<ops::min, Op, T> && builtin_in_category<T, integer, floating>) {
        return predefined_op::min;
    } else if constexpr (functor_of<std::plus, Op, T> && builtin_in_category<T, integer, floating, complex>) {
        return predefined_op::sum;
    } else if constexpr (functor_of<std::multiplies, Op, T> && builtin_in_category<T, integer, floating, complex>) {
        return predefined_op::prod;
    } else if constexpr (functor_of<std::logical_and, Op, T> && builtin_in_category<T, integer, logical>) {
        return predefined_op::land;
    } else if constexpr (functor_of<std::logical_or, Op, T> && builtin_in_category<T, integer, logical>) {
        return predefined_op::lor;
    } else if constexpr (functor_of<ops::logical_xor, Op, T> && builtin_in_category<T, integer, logical>) {
        return predefined_op::lxor;
    } else if constexpr (functor_of<std::bit_and, Op, T> && builtin_in_category<T, integer, byte>) {
        return predefined_op::band;
    } else if constexpr (functor_of<std::bit_or, Op, T> && builtin_in_category<T, integer, byte>) {
        return predefined_op::bor;
    } else if constexpr (functor_of<std::bit_xor, Op, T> && builtin_in_category<T, integer, byte>) {
        return predefined_op::bxor;
    } else {
        return predefined_op::none;
    }
}

} // namespace detail

/// The function object `Op` applied to elements of type `T` is a predefined MPI operation, e.g. `std::plus<>` on
/// `int` (`MPI_SUM`) or `ops::max<>` on `double` (`MPI_MAX`), ignoring cv- and reference qualifiers.
///
/// The functors are `std::plus`, `std::multiplies`, `std::logical_and`, `std::logical_or`, `std::bit_and`,
/// `std::bit_or`, `std::bit_xor` (also available under `ops::`) and `ops::max`, `ops::min`, `ops::logical_xor`, each
/// as `F<>` or `F<T>`. Which of them apply to which types follows the MPI standard; e.g. `std::bit_and<>` applies to
/// integers and `std::byte`, but not to `double`. Enumerations have no predefined operations. The operation itself is
/// `builtin_op_handle<Op, T>()`.
template <typename Op, typename T>
concept builtin_op =
    detail::predefined_op_of<std::remove_cvref_t<Op>, std::remove_cvref_t<T>>() != detail::predefined_op::none;

/// @brief The predefined MPI operation of the function object `Op` on elements of type `T`, e.g. `MPI_SUM` for
/// `std::plus<>` on `int`.
template <typename Op, typename T>
    requires builtin_op<Op, T>
[[nodiscard]] MPI_Op builtin_op_handle() noexcept {
    switch (detail::predefined_op_of<std::remove_cvref_t<Op>, std::remove_cvref_t<T>>()) {
        case detail::predefined_op::max:
            return MPI_MAX;
        case detail::predefined_op::min:
            return MPI_MIN;
        case detail::predefined_op::sum:
            return MPI_SUM;
        case detail::predefined_op::prod:
            return MPI_PROD;
        case detail::predefined_op::land:
            return MPI_LAND;
        case detail::predefined_op::lor:
            return MPI_LOR;
        case detail::predefined_op::lxor:
            return MPI_LXOR;
        case detail::predefined_op::band:
            return MPI_BAND;
        case detail::predefined_op::bor:
            return MPI_BOR;
        case detail::predefined_op::bxor:
            return MPI_BXOR;
        case detail::predefined_op::none:
            break;
    }
    return MPI_OP_NULL;
}

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
concept is_builtin_range_op = std::ranges::range<Buf> && builtin_op<Op, std::ranges::range_value_t<Buf>>;

} // namespace detail

// ──────────────────────────────────────────────────────────────────────────────
// as_mpi_op() — priority: op_traits > builtin handle > range builtin op
// ──────────────────────────────────────────────────────────────────────────────

/// @return The `MPI_Op` for `op`, obtained via `op_traits`.
template <typename Op, send_buffer SBuf, recv_buffer RBuf>
    requires detail::traits_has_op<Op, SBuf, RBuf>
constexpr MPI_Op as_mpi_op(Op const& op, SBuf const& sbuf, RBuf const& rbuf) {
    return op_traits<std::remove_cvref_t<Op>, SBuf, RBuf>::op(op, sbuf, rbuf);
}

/// @return The `MPI_Op` for `op`, which is (or wraps) a raw `MPI_Op` handle.
template <typename Op, send_buffer SBuf, recv_buffer RBuf>
    requires(!detail::traits_has_op<Op, SBuf, RBuf>) && detail::is_builtin_mpi_op<Op>
constexpr MPI_Op as_mpi_op(Op const& op, SBuf const&, RBuf const&) {
    return mpi::experimental::handle(op);
}

/// @return The predefined `MPI_Op` for a standard functor such as `std::plus<>`, given the buffers' element type.
template <typename Op, send_buffer SBuf, recv_buffer RBuf>
    requires(!detail::traits_has_op<Op, SBuf, RBuf>) && (!detail::is_builtin_mpi_op<Op>)
            && (detail::is_builtin_range_op<Op, SBuf> || detail::is_builtin_range_op<Op, RBuf>)
constexpr MPI_Op as_mpi_op(Op const&, SBuf const&, RBuf const&) {
    using range_buf_t = std::conditional_t<detail::is_builtin_range_op<Op, SBuf>, SBuf, RBuf>;
    return mpi::experimental::builtin_op_handle<Op, std::ranges::range_value_t<range_buf_t>>();
}

// ──────────────────────────────────────────────────────────────────────────────
// valid_op concept — checks if as_mpi_op is well-defined
// ──────────────────────────────────────────────────────────────────────────────

/// `Op` can be used as reduction operation on buffers of types `SBuf` and `RBuf`, i.e. `as_mpi_op` is defined.
template <typename Op, typename SBuf, typename RBuf>
concept valid_op =
    requires(Op const& op, SBuf const& sbuf, RBuf const& rbuf) { mpi::experimental::as_mpi_op(op, sbuf, rbuf); };

} // namespace mpi::experimental
