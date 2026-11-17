// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <complex>
#include <cstddef>
#include <type_traits>

#include <mpi.h>

/// @file
/// The mapping of C++ types to predefined MPI datatypes: the concept `builtin_mpi_type<T>`, the function
/// `builtin_datatype<T>()`, and `unpacked_bool`, a `bool` that can be stored in a contiguous container.

namespace mpi::experimental {

/// @brief A `bool` stored in a byte of its own, for containers of booleans that MPI can send.
///
/// `std::vector<bool>` packs its elements into bits, so it is not a contiguous range of `bool` and cannot be a buffer.
/// `std::vector<unpacked_bool>` stores one `bool` per element and is sent as `MPI_CXX_BOOL`. `unpacked_bool` converts
/// implicitly from and to `bool`.
///
/// @code
/// std::vector<unpacked_bool> flags(comm.size());
/// allgather(views::ref_single(unpacked_bool{converged}), flags, comm);
/// @endcode
class unpacked_bool {
public:
    /// @brief Holds `false`.
    constexpr unpacked_bool() noexcept = default;

    /// @brief Holds `value`.
    constexpr unpacked_bool(bool value) noexcept : _value(value) {}

    /// @return The value.
    constexpr operator bool() const noexcept {
        return _value;
    }

private:
    bool _value = false;
};

namespace detail {

// The groups of predefined datatypes that determine which predefined reduction operations apply (MPI 4.1, Section
// 6.9.2). `character` (MPI_CHAR, MPI_WCHAR) is in none of them.
enum class builtin_category { integer, floating, complex, logical, byte, character };

// The predefined datatype of T: `static MPI_Datatype datatype()` and `static constexpr builtin_category category`.
// Not defined for other types. Only exact types are mapped; cv-qualifiers are removed by the users of this trait.
template <typename T>
struct builtin_type_info {};

// The MPI datatype handles are not constant expressions in every implementation (in Open MPI, they are addresses of
// global objects), so they are returned by functions.

template <>
struct builtin_type_info<char> {
    static constexpr builtin_category category = builtin_category::character;
    static MPI_Datatype               datatype() noexcept {
        return MPI_CHAR;
    }
};

template <>
struct builtin_type_info<wchar_t> {
    static constexpr builtin_category category = builtin_category::character;
    static MPI_Datatype               datatype() noexcept {
        return MPI_WCHAR;
    }
};

template <>
struct builtin_type_info<signed char> {
    static constexpr builtin_category category = builtin_category::integer;
    static MPI_Datatype               datatype() noexcept {
        return MPI_SIGNED_CHAR;
    }
};

template <>
struct builtin_type_info<unsigned char> {
    static constexpr builtin_category category = builtin_category::integer;
    static MPI_Datatype               datatype() noexcept {
        return MPI_UNSIGNED_CHAR;
    }
};

template <>
struct builtin_type_info<short> {
    static constexpr builtin_category category = builtin_category::integer;
    static MPI_Datatype               datatype() noexcept {
        return MPI_SHORT;
    }
};

template <>
struct builtin_type_info<unsigned short> {
    static constexpr builtin_category category = builtin_category::integer;
    static MPI_Datatype               datatype() noexcept {
        return MPI_UNSIGNED_SHORT;
    }
};

template <>
struct builtin_type_info<int> {
    static constexpr builtin_category category = builtin_category::integer;
    static MPI_Datatype               datatype() noexcept {
        return MPI_INT;
    }
};

template <>
struct builtin_type_info<unsigned int> {
    static constexpr builtin_category category = builtin_category::integer;
    static MPI_Datatype               datatype() noexcept {
        return MPI_UNSIGNED;
    }
};

template <>
struct builtin_type_info<long> {
    static constexpr builtin_category category = builtin_category::integer;
    static MPI_Datatype               datatype() noexcept {
        return MPI_LONG;
    }
};

template <>
struct builtin_type_info<unsigned long> {
    static constexpr builtin_category category = builtin_category::integer;
    static MPI_Datatype               datatype() noexcept {
        return MPI_UNSIGNED_LONG;
    }
};

template <>
struct builtin_type_info<long long> {
    static constexpr builtin_category category = builtin_category::integer;
    static MPI_Datatype               datatype() noexcept {
        return MPI_LONG_LONG;
    }
};

template <>
struct builtin_type_info<unsigned long long> {
    static constexpr builtin_category category = builtin_category::integer;
    static MPI_Datatype               datatype() noexcept {
        return MPI_UNSIGNED_LONG_LONG;
    }
};

template <>
struct builtin_type_info<float> {
    static constexpr builtin_category category = builtin_category::floating;
    static MPI_Datatype               datatype() noexcept {
        return MPI_FLOAT;
    }
};

template <>
struct builtin_type_info<double> {
    static constexpr builtin_category category = builtin_category::floating;
    static MPI_Datatype               datatype() noexcept {
        return MPI_DOUBLE;
    }
};

template <>
struct builtin_type_info<long double> {
    static constexpr builtin_category category = builtin_category::floating;
    static MPI_Datatype               datatype() noexcept {
        return MPI_LONG_DOUBLE;
    }
};

template <>
struct builtin_type_info<bool> {
    static constexpr builtin_category category = builtin_category::logical;
    static MPI_Datatype               datatype() noexcept {
        return MPI_CXX_BOOL;
    }
};

template <>
struct builtin_type_info<unpacked_bool> {
    static constexpr builtin_category category = builtin_category::logical;
    static MPI_Datatype               datatype() noexcept {
        return MPI_CXX_BOOL;
    }
};

template <>
struct builtin_type_info<std::byte> {
    static constexpr builtin_category category = builtin_category::byte;
    static MPI_Datatype               datatype() noexcept {
        return MPI_BYTE;
    }
};

template <>
struct builtin_type_info<std::complex<float>> {
    static constexpr builtin_category category = builtin_category::complex;
    static MPI_Datatype               datatype() noexcept {
        return MPI_CXX_FLOAT_COMPLEX;
    }
};

template <>
struct builtin_type_info<std::complex<double>> {
    static constexpr builtin_category category = builtin_category::complex;
    static MPI_Datatype               datatype() noexcept {
        return MPI_CXX_DOUBLE_COMPLEX;
    }
};

template <>
struct builtin_type_info<std::complex<long double>> {
    static constexpr builtin_category category = builtin_category::complex;
    static MPI_Datatype               datatype() noexcept {
        return MPI_CXX_LONG_DOUBLE_COMPLEX;
    }
};

// T (without cv-qualifiers) is one of the types mapped above.
template <typename T>
concept arithmetic_builtin = requires { builtin_type_info<T>::datatype(); };

// An enumeration whose underlying type is mapped above, other than std::byte, which is mapped itself. It is sent with
// the datatype of its underlying type, but has no category: the predefined reduction operations do not apply to it.
template <typename T>
concept enum_builtin = std::is_enum_v<T> && !arithmetic_builtin<T> && arithmetic_builtin<std::underlying_type_t<T>>;

} // namespace detail

/// A type with a predefined MPI datatype, ignoring cv- and reference qualifiers: an arithmetic type such as `int`
/// (`MPI_INT`) or `double` (`MPI_DOUBLE`), `bool` and `unpacked_bool` (`MPI_CXX_BOOL`), `std::byte` (`MPI_BYTE`),
/// `std::complex<float>` etc., or an enumeration whose underlying type is one of these, which is sent with the datatype
/// of its underlying type. Contiguous ranges of such types are buffers without further adaption. The datatype itself is
/// `builtin_datatype<T>()`.
///
/// The predefined reduction operations, such as `std::plus<>` as `MPI_SUM`, apply to the arithmetic types only, not to
/// enumerations; see `builtin_op`.
template <typename T>
concept builtin_mpi_type =
    detail::arithmetic_builtin<std::remove_cvref_t<T>> || detail::enum_builtin<std::remove_cvref_t<T>>;

/// @brief The predefined MPI datatype of `T`, e.g. `MPI_DOUBLE` for `double`; for an enumeration, the datatype of its
/// underlying type.
///
/// Predefined datatypes are never committed or freed, so the result can be used anywhere a datatype is expected:
/// `type_vector(n, 1, n, builtin_datatype<double>())`.
template <builtin_mpi_type T>
[[nodiscard]] MPI_Datatype builtin_datatype() noexcept {
    using U = std::remove_cvref_t<T>;
    // std::byte is an enumeration with a datatype of its own, so the mapped types come first.
    if constexpr (detail::arithmetic_builtin<U>) {
        return detail::builtin_type_info<U>::datatype();
    } else {
        return detail::builtin_type_info<std::underlying_type_t<U>>::datatype();
    }
}

} // namespace mpi::experimental
