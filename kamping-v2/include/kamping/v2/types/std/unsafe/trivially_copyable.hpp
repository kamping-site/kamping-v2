// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <array>
#include <type_traits>
#include <utility>

#include "kamping/v2/types/as_struct.hpp"
#include "kamping/v2/types/datatype_traits.hpp"
#include "mpi/builtin_types.hpp"

/// @file
/// Describes every trivially copyable type that has no datatype otherwise as its bytes (`as_bytes`), including
/// padding.
///
/// Opt-in and unsafe: sending the padding is usually faster than skipping it, but only correct if all processes use the
/// same representation of the types. Builtin types, arrays, `std::pair` and `std::tuple` are left out, so this header
/// combines with the headers for pairs and tuples. A full specialization of `datatype_traits` for a type still takes
/// precedence; a partial specialization for a trivially copyable class template is ambiguous with this one.

namespace kamping::v2 {

namespace detail {
template <typename T>
struct is_std_array : std::false_type {};
template <typename E, std::size_t N>
struct is_std_array<std::array<E, N>> : std::true_type {};
} // namespace detail

/// @brief Every other trivially copyable class or union type is described as its bytes (`as_bytes`).
template <typename T>
    requires std::is_trivially_copyable_v<T> && (std::is_class_v<T> || std::is_union_v<T>)
             && (!mpi::experimental::builtin_mpi_type<T>) && (!detail::is_std_array<T>::value)
             && (!detail::is_pair_or_tuple<T>::value)
struct datatype_traits<T> : as_bytes<T> {};

} // namespace kamping::v2
