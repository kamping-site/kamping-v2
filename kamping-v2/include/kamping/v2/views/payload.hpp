// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <ranges>
#include <type_traits>

#include <kamping/types/builtin_types.hpp>
#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/handle.hpp"

/// @file
/// Structured send buffers, their payload element type, and the value-type channel.
///
/// Structured send buffers — a range of `(value, rank)` pairs (value_destination_pair_buffer), a
/// range of `(rank, message)` pairs (sparse_nested_send_buffer), or a range of messages indexed
/// by rank (nested_send_buffer) — carry a payload element that is distinct from the range's own
/// value_type. `payload_element_t<R>` resolves that payload type for any of these shapes, and
/// falls back to the plain element type for ordinary flat buffers.
///
/// `kamping::v2::value_type(buf)` is orthogonal to `mpi::experimental::type(buf)` ("the MPI type
/// of this buffer's elements"): it names the MPI type of the payload of a structured buffer. It
/// is not part of the core buffer contract (mpi/buffer.hpp), since only consumers of structured
/// buffers (flatten_v_view, the dstl algorithms) need it, and has its own customization point
/// (value_type_traits).

namespace kamping::v2 {

// ──────────────────────────────────────────────────────────────────────────────
// mpi_element_type_t — the buffer's own element type
// ──────────────────────────────────────────────────────────────────────────────

/// Resolves a buffer's own element type, always treating it as flat — unlike
/// `payload_element_t`, this never interprets `R` as a structured buffer.
/// Priority: `std::ranges::range_value_t<R>` for standard ranges, else a public `R::value_type`
/// member (kokkos_view, sycl_view, …).
///
/// The two differ when a flat buffer only looks structured by coincidence: for
/// `std::vector<std::pair<int, int>>`, the second field satisfies `rank`, so `payload_element_t`
/// takes it for a value_destination_pair_buffer and resolves to `int`. Hence the buffer-type
/// channel (`with_pool`, `with_auto_pool`) uses `mpi_element_type_t`, and only the value-type
/// channel (`with_value_pool`, `with_auto_value_pool`) uses `payload_element_t`.
template <typename R>
struct mpi_element_type {};

template <std::ranges::range R>
struct mpi_element_type<R> {
    using type = std::ranges::range_value_t<R>;
};

template <typename R>
    requires(!std::ranges::range<R>) && requires { typename R::value_type; }
struct mpi_element_type<R> {
    using type = typename R::value_type;
};

/// The buffer's own element type; see @ref mpi_element_type.
template <typename R>
using mpi_element_type_t = typename mpi_element_type<R>::type;

// ──────────────────────────────────────────────────────────────────────────────
// Messages: the per-destination slot of a nested or sparse send buffer.
//
// The structure concepts below only require a slot to be *some* message. What a consumer
// does with a message determines which capability it additionally requires:
//   - range_message:  iterate it element by element (flatten_v_view copies messages);
//   - buffer_message: access it in place via the buffer protocol (dstl::sparse_alltoall sends
//                     each message directly from its storage).
// Neither requires the message to know its MPI datatype: structured buffers carry that in the
// value-type channel (value_type() below), one datatype for all messages.
// ──────────────────────────────────────────────────────────────────────────────

/// A message that can be iterated element by element: a sized input range.
template <typename M>
concept range_message = std::ranges::input_range<M> && std::ranges::sized_range<M>;

/// A message accessible in place through the buffer protocol: a pointer to its data and an
/// element count (`mpi::experimental::ptr` / `count`). Contiguous ranges qualify, as do custom
/// buffers adapted via `buffer_traits` or `mpi_ptr()` / `mpi_count()` members.
template <typename M>
concept buffer_message = requires(M const& m) {
    { mpi::experimental::ptr(m) } -> std::convertible_to<void const*>;
    mpi::experimental::count(m);
};

/// The slot of a nested or sparse send buffer: a message with either capability.
template <typename M>
concept message = range_message<M> || buffer_message<M>;

/// Element type of a message: `range_value_t` for ranges, else a public `value_type` member.
/// Undefined for messages that expose neither; their datatype must then be attached
/// explicitly via the value-type channel (`views::with_value_type`).
template <typename M>
using message_element_t = mpi_element_type_t<std::remove_cvref_t<M>>;

// ──────────────────────────────────────────────────────────────────────────────
// Flattenable buffer shapes and their payload element type.
// ──────────────────────────────────────────────────────────────────────────────

/// Opt-out marker: true for a buffer type that is already flat and must never be interpreted as a
/// structured buffer, whatever its element type looks like. The structure concepts below match on
/// shape only, so e.g. a flat buffer of `std::pair<int, int>` would otherwise satisfy
/// value_destination_pair_buffer. flatten_v_view specializes this to true.
template <typename T>
inline constexpr bool already_flat_buffer = false;

/// A sized range of messages, indexed by destination rank.
template <typename R>
concept nested_send_buffer = !already_flat_buffer<std::remove_cvref_t<R>> && std::ranges::forward_range<R>
                             && std::ranges::sized_range<R> && message<std::ranges::range_value_t<R>>;

/// A non-range type with exactly two tuple elements, such as `std::pair`.
template <typename T>
concept pair_like = !std::ranges::range<T> && requires { std::tuple_size<T>::value; } && std::tuple_size<T>::value == 2;

/// A `(destination, message)` pair.
template <typename T>
concept destination_buffer_pair =
    pair_like<T> && mpi::experimental::rank<std::tuple_element_t<0, T>> && message<std::tuple_element_t<1, T>>;

/// A range of `(destination, message)` pairs.
template <typename R>
concept sparse_nested_send_buffer = !already_flat_buffer<std::remove_cvref_t<R>> && std::ranges::forward_range<R>
                                    && destination_buffer_pair<std::ranges::range_value_t<R>>;

/// The message type of a nested or sparse send buffer.
template <typename R>
struct nested_message {};

template <nested_send_buffer R>
struct nested_message<R> {
    using type = std::ranges::range_value_t<R>;
};

template <sparse_nested_send_buffer R>
struct nested_message<R> {
    using type = std::tuple_element_t<1, std::ranges::range_value_t<R>>;
};

/// The message type of a nested or sparse send buffer; see @ref nested_message.
template <typename R>
using nested_message_t = typename nested_message<R>::type;

/// A `(value, destination)` pair carrying a single element for one rank.
///
/// The value must not itself be a message: multiple elements for a rank are expressed as a
/// `(destination, message)` pair (destination_buffer_pair). A `(range, rank)` pair therefore
/// matches no structure concept.
template <typename T>
concept value_destination_pair =
    pair_like<T> && mpi::experimental::rank<std::tuple_element_t<1, T>> && !message<std::tuple_element_t<0, T>>;

/// A range of `(value, destination)` pairs.
template <typename R>
concept value_destination_pair_buffer = !already_flat_buffer<std::remove_cvref_t<R>> && std::ranges::forward_range<R>
                                        && value_destination_pair<std::ranges::range_value_t<R>>;

/// Any structured send buffer: nested, sparse or value-destination pairs. Purely structural;
/// see @ref copyable_send_buffer for what flatten_v_view can actually copy.
template <typename R>
concept flattenable_send_buffer =
    nested_send_buffer<R> || sparse_nested_send_buffer<R> || value_destination_pair_buffer<R>;

/// A structured send buffer whose payload can be copied element by element: value-destination
/// pairs, or nested / sparse buffers of range messages. Required by flatten_v_view.
template <typename R>
concept copyable_send_buffer =
    value_destination_pair_buffer<R>
    || ((nested_send_buffer<R> || sparse_nested_send_buffer<R>) && range_message<nested_message_t<R>>);

/// The payload element type of a structured send buffer, or the element type of a flat buffer.
template <typename R>
struct payload_element {};

template <typename R>
    requires(nested_send_buffer<R> || sparse_nested_send_buffer<R>)
            && requires { typename message_element_t<nested_message_t<R>>; }
struct payload_element<R> {
    using type = message_element_t<nested_message_t<R>>;
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
/// is its payload.
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

/// The payload element type of `R`; see @ref payload_element.
template <typename R>
using payload_element_t = typename payload_element<R>::type;

// ──────────────────────────────────────────────────────────────────────────────
// value_type() — the value-type channel (see file documentation).
// Priority: value_type_traits > mpi_value_type() > builtin payload type deduction
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

/// @return The MPI datatype of the payload of `t`, obtained via `value_type_traits<T>::value_type(t)`.
template <typename T>
    requires detail::traits_has_value_type<std::remove_cvref_t<T>>
constexpr auto value_type(T&& t) {
    return value_type_traits<std::remove_cvref_t<T>>::value_type(t);
}

/// @return The MPI datatype of the payload of `t`, obtained via `t.mpi_value_type()`.
template <typename T>
    requires(!detail::traits_has_value_type<std::remove_cvref_t<T>>) && detail::has_value_type_member<T>
constexpr auto value_type(T&& t) {
    return t.mpi_value_type();
}

/// @return The predefined MPI datatype of the payload element type of `T`.
template <typename T>
    requires(!detail::traits_has_value_type<std::remove_cvref_t<T>>) && (!detail::has_value_type_member<T>)
            && detail::payload_element_is_builtin<T>
constexpr auto value_type(T&& /* t */) {
    return kamping::types::builtin_type<payload_element_t<T>>::data_type();
}

/// `value_type()` is defined for `T`.
template <typename T>
concept has_mpi_value_type = requires(T const& t) {
    { kamping::v2::value_type(t) } -> std::convertible_to<MPI_Datatype>;
};

} // namespace kamping::v2
