// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <type_traits>

namespace kamping::v2 {
/// Tags selecting the MPI send mode of `send` and `isend`, e.g. `v2::send(send_mode::sync, buf, dest)`.
namespace send_mode {
struct standard_t {};                   ///< Standard mode (`MPI_Send`).
struct buffered_t {};                   ///< Buffered mode (`MPI_Bsend`).
struct sync_t {};                       ///< Synchronous mode (`MPI_Ssend`).
struct ready_t {};                      ///< Ready mode (`MPI_Rsend`).
inline constexpr standard_t standard{}; ///< Standard mode (`MPI_Send`).
inline constexpr buffered_t buffered{}; ///< Buffered mode (`MPI_Bsend`).
inline constexpr sync_t     sync{};     ///< Synchronous mode (`MPI_Ssend`).
inline constexpr ready_t    ready{};    ///< Ready mode (`MPI_Rsend`).
} // namespace send_mode

/// One of the @ref send_mode tags.
template <typename T>
concept is_send_mode = std::same_as<std::remove_cvref_t<T>, send_mode::standard_t>
                       || std::same_as<std::remove_cvref_t<T>, send_mode::buffered_t>
                       || std::same_as<std::remove_cvref_t<T>, send_mode::sync_t>
                       || std::same_as<std::remove_cvref_t<T>, send_mode::ready_t>;

} // namespace kamping::v2
