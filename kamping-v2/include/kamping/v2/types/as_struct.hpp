// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

#include <mpi.h>

#ifdef KAMPING_V2_ENABLE_REFLECTION
    #include <boost/pfr.hpp>
#endif

#include "kamping/v2/type_constructors.hpp"
#include "kamping/v2/types/datatype_traits.hpp"
#include "mpi/datatype.hpp"

/// @file
/// `as_struct`, the base for a `datatype_traits` specialization that derives the datatype of a struct from its fields.

namespace kamping::v2 {

/// @brief Tag for overriding whether Boost.PFR considers a type reflectable for `as_struct`, by specializing
/// `boost::pfr::is_reflectable<T, kamping::v2::as_struct_tag>`.
struct as_struct_tag {};

namespace detail {
template <typename T>
struct is_pair_or_tuple : std::false_type {};
template <typename A, typename B>
struct is_pair_or_tuple<std::pair<A, B>> : std::true_type {};
template <typename... Ts>
struct is_pair_or_tuple<std::tuple<Ts...>> : std::true_type {};

// T's fields can be visited: a std::pair or std::tuple, or, with reflection enabled, an aggregate that Boost.PFR can
// reflect.
template <typename T>
concept struct_like = is_pair_or_tuple<T>::value
#ifdef KAMPING_V2_ENABLE_REFLECTION
                      || boost::pfr::is_implicitly_reflectable<T, as_struct_tag>::value
#endif
    ;

// Calls f with each field of t, in declaration order.
template <struct_like T, typename F>
void for_each_field(T const& t, F&& f) {
    if constexpr (is_pair_or_tuple<T>::value) {
        std::apply([&](auto const&... fields) { (f(fields), ...); }, t);
    } else {
#ifdef KAMPING_V2_ENABLE_REFLECTION
        boost::pfr::for_each_field(t, f);
#endif
    }
}
} // namespace detail

/// @brief Base for a `datatype_traits` specialization that describes a struct by its fields (`MPI_Type_create_struct`).
///
/// The fields are found by reflection: those of a `std::pair` or `std::tuple`, or, if KaMPIng-v2 is built with
/// `KAMPING_V2_ENABLE_REFLECTION` (the default), those of an aggregate, using Boost.PFR. Every field needs a
/// datatype (`has_datatype`), and the datatype skips the padding between the fields. `T` must be default-constructible:
/// the field offsets are taken from a default-constructed object.
///
/// @code
/// struct point { int id; double x, y; };
/// template <>
/// struct kamping::v2::datatype_traits<point> : kamping::v2::as_struct<point> {};
/// @endcode
///
/// For a class that cannot be reflected, e.g. with constructors, private members or base classes, or of which only
/// some members are sent, write the specialization with a `struct_builder`. To send the padding as well, which is
/// usually faster, use `as_bytes`.
template <typename T>
    requires std::default_initializable<T> && detail::struct_like<T>
struct as_struct {
    /// @return The datatype of `T`, owned and not committed, resized to `sizeof(T)`.
    /// @throws mpi_error if MPI fails to create it.
    static mpi::experimental::uncommitted_datatype datatype() {
        T const           sample{};
        struct_builder<T> builder(sample);
        detail::for_each_field(sample, [&](auto const& field) {
            using field_type = std::remove_cvref_t<decltype(field)>;
            static_assert(
                has_datatype<field_type>,
                "as_struct: every field needs a datatype. Specialize kamping::v2::datatype_traits for the field type, "
                "or include kamping/v2/types/std/... for a std::pair or std::tuple field."
            );
            builder.field_of_sample(field, detail::datatype_of<field_type>());
        });
        return builder.build();
    }
};

} // namespace kamping::v2
