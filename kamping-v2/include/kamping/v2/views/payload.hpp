// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <ranges>
#include <type_traits>

#include <kamping/types/builtin_types.hpp>
#include <mpi.h>

#include "mpi/handle.hpp"

/// @file
/// Shared payload resolver + the value-type channel.
///
/// Structured send buffers — a range of `(value, rank)` pairs (value_destination_pair), a
/// range of `(rank, inner_range)` pairs (sparse_nested_send_buffer), or a plain range-of-ranges
/// (nested_send_buffer) — carry a payload element that is distinct from the range's own
/// value_type. `payload_element_t<R>` resolves that payload type for any of these shapes, and
/// falls back to the plain range/value_type element for ordinary flat buffers, subsuming the
/// old (weaker) `mpi_element_type_t` in type_pool.hpp.
///
/// `kamping::v2::value_type(buf)` is new vocabulary orthogonal to `mpi::experimental::type(buf)`
/// ("the MPI type of *this buffer's elements*"): it names the MPI type of the *payload / value
/// slot* of a structured buffer. It deliberately does NOT join the mpi::experimental:: buffer
/// protocol (include/mpi/buffer.hpp) — it is only meaningful to v2 machinery (flatten_v_view,
/// dstl::request_reply, …), so it lives here in the v2 layer with its own non-intrusive
/// customization point (value_type_traits), separate from mpi::experimental::buffer_traits.

namespace kamping::v2 {

// ──────────────────────────────────────────────────────────────────────────────
// Flattenable buffer shapes and their payload element type.
// ──────────────────────────────────────────────────────────────────────────────

template <typename R>
concept nested_send_buffer = std::ranges::forward_range<R> && std::ranges::sized_range<R>
                             && std::ranges::input_range<std::ranges::range_value_t<R>>
                             && std::ranges::sized_range<std::ranges::range_value_t<R>>;

template <typename T>
concept pair_like = requires(T t) { std::tuple_size<T>::value == 2; };

template <typename T>
concept destination_buffer_pair =
    pair_like<T> && mpi::experimental::rank<std::tuple_element_t<0, T>>
    && std::ranges::input_range<std::tuple_element_t<1, T>> && std::ranges::sized_range<std::tuple_element_t<1, T>>;

template <typename R>
concept sparse_nested_send_buffer =
    std::ranges::forward_range<R> && destination_buffer_pair<std::ranges::range_value_t<R>>;

// The value slot (element 0) must not itself be a range. A range-valued payload means
// "multiple elements for this rank" and belongs to the buffer path
// (destination_buffer_pair), not the single-value path. Without this exclusion a
// (range, rank) pair would match here and the whole range would be copied into a single
// element slot. Such a pair now matches no flattenable concept and fails to compile.
template <typename T>
concept value_destination_pair = pair_like<T> && mpi::experimental::rank<std::tuple_element_t<1, T>>
                                 && !std::ranges::input_range<std::tuple_element_t<0, T>>;

template <typename R>
concept value_destination_pair_buffer =
    std::ranges::forward_range<R> && value_destination_pair<std::ranges::range_value_t<R>>;

template <typename R>
concept flattenable_send_buffer =
    nested_send_buffer<R> || sparse_nested_send_buffer<R> || value_destination_pair_buffer<R>;

template <typename R>
struct payload_element {};

template <nested_send_buffer R>
struct payload_element<R> {
    using type = std::ranges::range_value_t<std::ranges::range_value_t<R>>;
};

template <sparse_nested_send_buffer R>
struct payload_element<R> {
    using type = std::ranges::range_value_t<std::tuple_element_t<1, std::ranges::range_value_t<R>>>;
};

template <value_destination_pair_buffer R>
struct payload_element<R> {
    // remove_cvref: the value slot may be a reference — e.g. a (value, rank) pair built
    // lazily to carry the payload by reference and avoid a copy — but the MPI element type
    // must be a plain value.
    using type = std::remove_cvref_t<std::tuple_element_t<0, std::ranges::range_value_t<R>>>;
};

/// Flat fallback: a buffer with no distinct payload (a plain range, or a custom buffer wrapper
/// exposing a public `value_type` member, e.g. kokkos_view / sycl_view) — its own element type
/// *is* its payload. Subsumes the old (weaker) `mpi_element_type_t` from type_pool.hpp.
template <typename R>
    requires(!flattenable_send_buffer<R>) && std::ranges::range<R>
struct payload_element<R> {
    using type = std::ranges::range_value_t<R>;
};

template <typename R>
    requires(!flattenable_send_buffer<R>) && (!std::ranges::range<R>) && requires { typename R::value_type; }
struct payload_element<R> {
    using type = typename R::value_type;
};

template <typename R>
using payload_element_t = typename payload_element<R>::type;

// ──────────────────────────────────────────────────────────────────────────────
// value_type() — the payload/value-type channel. New vocabulary, NOT part of the
// mpi::experimental:: buffer protocol (see file header). Three-tier dispatch mirroring
// mpi::experimental::type(), but with its own tier-1 customization point.
// ──────────────────────────────────────────────────────────────────────────────

/// Non-intrusive customization point for the value-type channel, analogous to
/// mpi::experimental::buffer_traits but scoped to the v2 layer.
template <typename T>
struct value_type_traits {};

namespace detail {

template <typename T>
concept traits_has_value_type = requires(T const& t) {
    { value_type_traits<T>::value_type(t) } -> std::convertible_to<MPI_Datatype>;
};

template <typename T>
concept has_value_type_member = requires(T const& t) {
    { t.mpi_value_type() } -> std::convertible_to<MPI_Datatype>;
};

template <typename T>
concept has_payload_element = requires { typename payload_element_t<T>; };

template <typename T>
concept payload_element_is_builtin = has_payload_element<T> && kamping::types::is_builtin_type_v<payload_element_t<T>>;

} // namespace detail

template <typename T>
    requires detail::traits_has_value_type<std::remove_cvref_t<T>>
constexpr auto value_type(T&& t) {
    return value_type_traits<std::remove_cvref_t<T>>::value_type(t);
}

template <typename T>
    requires(!detail::traits_has_value_type<std::remove_cvref_t<T>>) && detail::has_value_type_member<T>
constexpr auto value_type(T&& t) {
    return t.mpi_value_type();
}

template <typename T>
    requires(!detail::traits_has_value_type<std::remove_cvref_t<T>>) && (!detail::has_value_type_member<T>)
            && detail::payload_element_is_builtin<T>
constexpr auto value_type(T&& /* t */) {
    return kamping::types::builtin_type<payload_element_t<T>>::data_type();
}

template <typename T>
concept has_mpi_value_type = requires(T const& t) {
    { kamping::v2::value_type(t) } -> std::convertible_to<MPI_Datatype>;
};

} // namespace kamping::v2
