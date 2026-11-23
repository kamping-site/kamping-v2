// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <bit>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <mpi.h>

/// @file
/// The mapping of C++ types to predefined MPI datatypes: the concept `builtin_mpi_type<T>`, the function
/// `builtin_datatype<T>()`, the customization point `representation_traits<T>` for types that are sent as an
/// arithmetic type (such as `std::chrono::duration`), and `unpacked_bool`, a `bool` that can be stored in a contiguous
/// container.

namespace mpi::experimental {

/// @brief Declares that a `T` is sent with the predefined datatype of the arithmetic type `type`, because its bytes
/// are a `type`.
///
/// Specialize it for wrappers around a single arithmetic value, such as a strong typedef:
///
/// @code
/// struct meters {
///     double value;
///     friend meters operator+(meters a, meters b) { return {a.value + b.value}; }
///     friend auto   operator<=>(meters, meters) = default;
/// };
/// template <>
/// struct mpi::experimental::representation_traits<meters> {
///     using type = double;
/// };
/// @endcode
///
/// A specialization promises that the object representation of a `T` is that of its `type`, so that MPI can read and
/// write a `T` as a `type`. It only takes effect if `type` is an arithmetic builtin type (`int`, `double`, ...; not an
/// enumeration or another represented type) and `T` is trivially copyable with the size and alignment of `type`;
/// otherwise `T` is not a builtin type.
///
/// A predefined reduction operation applies to `T` if its function object computes a `T` from two `T`s and MPI
/// defines the operation for `type`: `std::plus<>` and `ops::max<>` on `meters` become `MPI_SUM` and `MPI_MAX`, but
/// `std::multiplies<>` does not apply, since `meters * meters` is not defined. The specialization also promises that
/// these functors compute on a `T` what the MPI operation computes on its `type`.
///
/// `std::chrono::duration`s with an arithmetic `rep` and `unpacked_bool` are specialized by the library.
template <typename T>
struct representation_traits {};

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

// The Unicode character types have the representation of unsigned integers ([basic.fundamental]), but MPI has no
// character datatypes for them. Like char, they get no predefined operations.

template <>
struct builtin_type_info<char8_t> {
    static constexpr builtin_category category = builtin_category::character;
    static MPI_Datatype               datatype() noexcept {
        return MPI_UNSIGNED_CHAR;
    }
};

template <>
struct builtin_type_info<char16_t> {
    static constexpr builtin_category category = builtin_category::character;
    static MPI_Datatype               datatype() noexcept {
        return builtin_type_info<std::uint_least16_t>::datatype();
    }
};

template <>
struct builtin_type_info<char32_t> {
    static constexpr builtin_category category = builtin_category::character;
    static MPI_Datatype               datatype() noexcept {
        return builtin_type_info<std::uint_least32_t>::datatype();
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

// A class that representation_traits declares to be sent as an arithmetic builtin type, and whose layout allows it:
// trivially copyable, so that MPI copying its bytes is copying the object, with the size and alignment of that type.
template <typename T>
concept represented_builtin =
    std::is_class_v<T> && requires { typename representation_traits<T>::type; }
    && arithmetic_builtin<typename representation_traits<T>::type> && std::is_trivially_copyable_v<T>
    && sizeof(T) == sizeof(typename representation_traits<T>::type)
    && alignof(T) == alignof(typename representation_traits<T>::type);

// The bytes of a D constructed from a Rep are that Rep, checked for a few values: zero, one, minus one and, for
// arithmetic types, the extremes. The standard specifies duration with a single member of type rep only for
// exposition, so this is checked rather than assumed.
template <typename D, typename Rep>
consteval bool constructs_from_its_representation() {
    if constexpr (!std::is_trivially_copyable_v<D> || sizeof(D) != sizeof(Rep) || alignof(D) != alignof(Rep)) {
        return false;
    } else {
        auto const round_trips = [](Rep const value) {
            return std::bit_cast<Rep>(D{value}) == value;
        };
        bool result = round_trips(Rep(0)) && round_trips(Rep(1)) && round_trips(static_cast<Rep>(-1));
        if constexpr (std::is_arithmetic_v<Rep>) {
            result = result && round_trips(std::numeric_limits<Rep>::max())
                     && round_trips(std::numeric_limits<Rep>::lowest());
        }
        return result;
    }
}

} // namespace detail

/// @brief A `std::chrono::duration` is sent as its `rep`, e.g. `std::chrono::nanoseconds` as a `long long`
/// (`MPI_LONG_LONG`) and `clock::duration` as a `double`.
///
/// Predefined reduction operations apply as far as they are defined on durations: `ops::max<>`, `ops::min<>` and
/// `std::plus<>`. The `rep` must be an arithmetic builtin type, and the standard library must store a duration as its
/// `rep`, which is checked at compile time. The period is not sent: the receiving process must use the same duration
/// type.
template <typename Rep, typename Period>
    requires detail::arithmetic_builtin<Rep>
             && (detail::constructs_from_its_representation<std::chrono::duration<Rep, Period>, Rep>())
struct representation_traits<std::chrono::duration<Rep, Period>> {
    /// The arithmetic type a duration is sent as.
    using type = Rep;
};

/// @brief `unpacked_bool` is sent as a `bool` (`MPI_CXX_BOOL`).
template <>
struct representation_traits<unpacked_bool> {
    /// The arithmetic type an `unpacked_bool` is sent as.
    using type = bool;
};

/// A type with a predefined MPI datatype, ignoring cv- and reference qualifiers:
///
/// - an arithmetic type such as `int` (`MPI_INT`) or `double` (`MPI_DOUBLE`), `bool` (`MPI_CXX_BOOL`), `std::byte`
///   (`MPI_BYTE`), `std::complex<float>` etc.; `char8_t`, `char16_t` and `char32_t` are sent as unsigned integers of
///   their size;
/// - an enumeration whose underlying type is one of these, sent with the datatype of its underlying type;
/// - a class that `representation_traits` declares to be sent as one of these, such as `std::chrono::duration` and
///   `unpacked_bool`.
///
/// Contiguous ranges of such types are buffers without further adaption. The datatype itself is
/// `builtin_datatype<T>()`.
///
/// The predefined reduction operations, such as `std::plus<>` as `MPI_SUM`, apply to the arithmetic types and, as far
/// as their function objects are defined on them, to represented classes, but not to enumerations or character types;
/// see `builtin_op`.
template <typename T>
concept builtin_mpi_type =
    detail::arithmetic_builtin<std::remove_cvref_t<T>> || detail::enum_builtin<std::remove_cvref_t<T>>
    || detail::represented_builtin<std::remove_cvref_t<T>>;

/// @brief The predefined MPI datatype of `T`, e.g. `MPI_DOUBLE` for `double`; for an enumeration, the datatype of its
/// underlying type; for a class declared by `representation_traits`, the datatype of its representation.
///
/// Predefined datatypes are never committed or freed, so the result can be used anywhere a datatype is expected:
/// `type_vector(n, 1, n, builtin_datatype<double>())`.
template <builtin_mpi_type T>
[[nodiscard]] MPI_Datatype builtin_datatype() noexcept {
    using U = std::remove_cvref_t<T>;
    // std::byte is an enumeration with a datatype of its own, so the mapped types come first.
    if constexpr (detail::arithmetic_builtin<U>) {
        return detail::builtin_type_info<U>::datatype();
    } else if constexpr (detail::enum_builtin<U>) {
        return detail::builtin_type_info<std::underlying_type_t<U>>::datatype();
    } else {
        return detail::builtin_type_info<typename representation_traits<U>::type>::datatype();
    }
}

} // namespace mpi::experimental
