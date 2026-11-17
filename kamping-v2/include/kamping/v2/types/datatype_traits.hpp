// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <array>
#include <climits>
#include <concepts>
#include <cstddef>
#include <type_traits>

#include <mpi.h>

#include "mpi/builtin_types.hpp"
#include "mpi/datatype.hpp"
#include "mpi/type_constructors.hpp"

/// @file
/// The description of a C++ type as an MPI datatype: the customization point `datatype_traits<T>`, the concept
/// `has_datatype<T>`, and the bases `as_array` and `as_bytes` for specializations. `as_struct`, which derives the
/// datatype of a struct from its fields, is in kamping/v2/types/as_struct.hpp.

namespace kamping::v2 {

/// @brief Describes the C++ type `T` as an MPI datatype.
///
/// Specialize it for your own types. A specialization provides
///
/// @code
/// static mpi::experimental::uncommitted_datatype datatype();
/// @endcode
///
/// which returns a new datatype describing one `T`, with an extent of `sizeof(T)`, so that arrays of `T` work. Whoever
/// calls it owns the result; a `type_pool` commits it on first use and frees it with the pool. The bases `as_struct`,
/// `as_array` and `as_bytes` provide `datatype()` for the common cases:
///
/// @code
/// template <>
/// struct kamping::v2::datatype_traits<point> : kamping::v2::as_struct<point> {};
///
/// template <>
/// struct kamping::v2::datatype_traits<particle> {
///     static mpi::experimental::uncommitted_datatype datatype() {
///         return kamping::v2::struct_builder<particle>{}
///             .field(&particle::id)
///             .field(&particle::where, kamping::v2::datatype_traits<cell>::datatype())
///             .build();
///     }
/// };
/// @endcode
///
/// Builtin types such as `int` and enumerations have predefined datatypes and need no specialization (see
/// `builtin_mpi_type`). C arrays and `std::array` of types with a datatype are described as `as_array`. `std::pair`
/// and `std::tuple` are described by including one of the headers in kamping/v2/types/std/.
///
/// The library never uses a specialization implicitly: a type is sent with its derived datatype only through a
/// `type_pool`, e.g. with `views::with_auto_pool`.
template <typename T>
struct datatype_traits {};

namespace detail {
template <typename T>
concept traits_have_datatype = requires {
    { datatype_traits<T>::datatype() } -> std::same_as<mpi::experimental::uncommitted_datatype>;
};
} // namespace detail

/// `T` has an MPI datatype, ignoring cv- and reference qualifiers: it is a builtin type (`builtin_mpi_type`), whose
/// datatype is predefined, or `datatype_traits<T>` describes it.
template <typename T>
concept has_datatype = mpi::experimental::builtin_mpi_type<T> || detail::traits_have_datatype<std::remove_cvref_t<T>>;

namespace detail {
/// The datatype of one `T`, for composing the datatype of a type that contains a `T`: the predefined `MPI_Datatype` of
/// a builtin type, or a new `uncommitted_datatype` from `datatype_traits<T>`. Both can be passed as the base
/// datatype of a type constructor or a `struct_builder` field.
template <has_datatype T>
auto datatype_of() {
    using U = std::remove_cvref_t<T>;
    if constexpr (mpi::experimental::builtin_mpi_type<U>) {
        return mpi::experimental::builtin_datatype<U>();
    } else {
        return datatype_traits<U>::datatype();
    }
}
} // namespace detail

/// @brief Base for a `datatype_traits` specialization that describes a type as `N` consecutive elements of type `E`
/// (`MPI_Type_contiguous`).
///
/// `E` needs a datatype. Used for C arrays and `std::array`, and for classes that consist of `N` elements of `E`:
///
/// @code
/// struct vec4 { float x, y, z, w; };
/// template <>
/// struct kamping::v2::datatype_traits<vec4> : kamping::v2::as_array<float, 4> {};
/// @endcode
template <typename E, std::size_t N>
    requires has_datatype<E> && (N > 0) && (N <= static_cast<std::size_t>(INT_MAX))
struct as_array {
    /// @return The datatype of `N` elements of `E`, owned and not committed.
    /// @throws mpi_error if MPI fails to create it.
    static mpi::experimental::uncommitted_datatype datatype() {
        return mpi::experimental::type_contiguous(static_cast<int>(N), detail::datatype_of<E>());
    }
};

/// @brief Base for a `datatype_traits` specialization that describes a type as its `sizeof(T)` bytes (`MPI_BYTE`).
///
/// The bytes include the padding between and after the members, so the type is sent as one block of memory. This is
/// usually faster than a struct datatype that skips the gaps, which MPI implementations pack and unpack. It is only
/// correct if all processes use the same representation of `T`. `T` must be trivially copy-constructible and trivially
/// destructible, so that its bytes are its value; unlike trivially copyable, this admits `std::pair<int, int>`, whose
/// assignment is not trivial.
///
/// @code
/// template <>
/// struct kamping::v2::datatype_traits<particle> : kamping::v2::as_bytes<particle> {};
/// @endcode
template <typename T>
    requires std::is_trivially_copy_constructible_v<T> && std::is_trivially_destructible_v<T>
             && (sizeof(T) <= static_cast<std::size_t>(INT_MAX))
struct as_bytes {
    /// @return The datatype of `sizeof(T)` bytes, owned and not committed.
    /// @throws mpi_error if MPI fails to create it.
    static mpi::experimental::uncommitted_datatype datatype() {
        return mpi::experimental::type_contiguous(static_cast<int>(sizeof(T)), MPI_BYTE);
    }
};

/// @brief C arrays of types with a datatype are described as `as_array`.
template <typename E, std::size_t N>
    requires has_datatype<E> && (N <= static_cast<std::size_t>(INT_MAX))
struct datatype_traits<E[N]> : as_array<E, N> {};

/// @brief `std::array`s of types with a datatype are described as `as_array`.
template <typename E, std::size_t N>
    requires has_datatype<E> && (N > 0) && (N <= static_cast<std::size_t>(INT_MAX))
struct datatype_traits<std::array<E, N>> : as_array<E, N> {};

} // namespace kamping::v2
