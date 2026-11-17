// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <type_traits>

#include <mpi.h>

#include "mpi/handle.hpp"

/// @file
/// Typed versions of MPI's special input values (`MPI_ANY_SOURCE`, `MPI_ANY_TAG`, `MPI_PROC_NULL`, `MPI_UNDEFINED`),
/// and the role concepts that use them to reject erroneous arguments at compile time.
///
/// Each special value is an empty type that returns its MPI value from a `static constexpr` member function named
/// after its role (`mpi_rank()`, `mpi_tag()`), so `any_tag` cannot be passed as a rank. The type shows up in the
/// instantiated wrappers, and at `-O0` the member function is emitted as a named function, which lets verification
/// tools recognize the value. With optimization, it inlines to the literal, as with raw MPI. The raw macros are still
/// accepted wherever a rank or tag is.

namespace mpi::experimental {

/// @brief Type of @ref any_source.
struct any_source_t {
    /// @return `MPI_ANY_SOURCE`.
    static constexpr int mpi_rank() noexcept {
        return MPI_ANY_SOURCE;
    }
};

/// @brief Type of @ref any_tag.
struct any_tag_t {
    /// @return `MPI_ANY_TAG`.
    static constexpr int mpi_tag() noexcept {
        return MPI_ANY_TAG;
    }
};

/// @brief Type of @ref proc_null.
struct proc_null_t {
    /// @return `MPI_PROC_NULL`.
    static constexpr int mpi_rank() noexcept {
        return MPI_PROC_NULL;
    }
};

/// @brief Type of @ref undefined.
struct undefined_t {
    /// @return `MPI_UNDEFINED`.
    static constexpr int mpi_value() noexcept {
        return MPI_UNDEFINED;
    }
};

/// @brief Receive from any source (`MPI_ANY_SOURCE`). Not a valid destination.
inline constexpr any_source_t any_source{};

/// @brief Receive with any tag (`MPI_ANY_TAG`). Not a valid tag for a sent message.
inline constexpr any_tag_t any_tag{};

/// @brief The null process (`MPI_PROC_NULL`): a send to it or a receive from it completes immediately and transfers
/// nothing.
inline constexpr proc_null_t proc_null{};

/// @brief No color for `split` / no split type for `split_type` (`MPI_UNDEFINED`): the calling process is in none of
/// the new communicators and gets a null communicator.
inline constexpr undefined_t undefined{};

/// @brief The rank a message is received from: any rank, including `any_source` and `proc_null`.
template <typename T>
concept sender = rank<T>;

/// @brief The rank a message is sent to: any rank, including `proc_null`, but not `any_source`.
template <typename T>
concept destination = rank<T> && !std::same_as<std::remove_cvref_t<T>, any_source_t>;

/// @brief The tag a message is sent with: any tag except `any_tag`, which is only valid on the receiving side.
template <typename T>
concept message_tag = tag<T> && !std::same_as<std::remove_cvref_t<T>, any_tag_t>;

} // namespace mpi::experimental
