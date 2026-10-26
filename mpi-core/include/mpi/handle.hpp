// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include "mpi/range_concepts.hpp"

namespace mpi::experimental {

namespace detail {
// C++20 replacement for is_scoped_enum_v (C++23).
template <typename T, bool = std::is_enum_v<T>>
struct is_scoped_enum_impl : std::false_type {};
template <typename T>
struct is_scoped_enum_impl<T, true> : std::bool_constant<!std::is_convertible_v<T, std::underlying_type_t<T>>> {};
} // namespace detail

template <typename T>
inline constexpr bool is_scoped_enum_v = detail::is_scoped_enum_impl<T>::value;

template <typename T>
concept builtin_handle = std::same_as<T, MPI_Comm> || std::same_as<T, MPI_Datatype> || std::same_as<T, MPI_Request>
                         || std::same_as<T, MPI_Status> || std::same_as<T, MPI_Message> || std::same_as<T, MPI_Op>
                         || std::same_as<T, MPI_Info> || std::same_as<T, MPI_Group>;

template <typename T>
concept ptr_to_builtin_handle = std::is_pointer_v<T> && !std::is_const_v<T> && builtin_handle<std::remove_pointer_t<T>>;

/// A writable contiguous range of a builtin MPI handle type (e.g. `std::vector<MPI_Request>`). Not required to be
/// sized: whether a size is needed depends on the argument it is passed as (e.g. a request array sets the count).
template <typename T>
concept contiguous_handle_range = std::ranges::contiguous_range<T> && requires(T& t) {
    { std::ranges::data(t) } -> ptr_to_builtin_handle;
};

/// Writable storage of exactly `HandleType` that refers to the caller's memory (see `output_range_of`).
template <typename T, typename HandleType>
concept contiguous_handle_range_of = builtin_handle<HandleType> && output_range_of<T, HandleType>;

namespace detail {
template <typename T>
concept has_mpi_handle_member = requires(T const& t) {
    { t.mpi_handle() } -> builtin_handle;
};

template <typename T>
concept has_mpi_handle_ptr_member = requires(T& t) {
    { t.mpi_handle_ptr() } -> ptr_to_builtin_handle;
};

template <typename T>
concept has_mpi_handle_range_member = requires(T& t) {
    { t.mpi_handle_range() } -> contiguous_handle_range;
};
} // namespace detail

// ──────────────────────────────────────────────────────────────────────────────
// native_handle_traits — trait class for non-intrusive MPI handle customization.
//
// Specialize native_handle_traits<T> for types you don't own. Only implement
// the members you need; unimplemented members fall back to the corresponding
// mpi_native_handle() / mpi_handle_ptr() member functions.
//
//   template <>
//   struct native_handle_traits<MyComm> {
//       static MPI_Comm handle(MyComm const& c)  { return c.get_comm(); }
//       static MPI_Comm* handle_ptr(MyComm& c)   { return c.get_comm_ptr(); }
//   };
// ──────────────────────────────────────────────────────────────────────────────
template <typename T>
struct handle_traits {};

/// Trait class for non-intrusively exposing the contiguous handle storage of a type you don't own.
/// Specialize with `static <contiguous_handle_range> handle_range(T& t)`; return a reference or a borrowed view
/// (e.g. std::span) into `t`, never a copy.
template <typename T>
struct handle_range_traits {};

namespace detail {
template <typename T>
concept handle_traits_has_handle = requires(T const& t) {
    { handle_traits<T>::handle(t) } -> builtin_handle;
};

template <typename T>
concept handle_traits_has_handle_ptr = requires(T& t) {
    { handle_traits<T>::handle_ptr(t) } -> ptr_to_builtin_handle;
};

template <typename T>
concept handle_range_traits_has_handle_range = requires(T& t) {
    { handle_range_traits<T>::handle_range(t) } -> contiguous_handle_range;
};
} // namespace detail

// ──────────────────────────────────────────────────────────────────────────────
// native_handle() dispatch — priority: native_handle_traits > member > builtin passthrough
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
    requires detail::handle_traits_has_handle<std::remove_cvref_t<T>>
constexpr auto handle(T const& t) {
    return handle_traits<std::remove_cvref_t<T>>::handle(t);
}

template <typename T>
    requires(!detail::handle_traits_has_handle<std::remove_cvref_t<T>>) && detail::has_mpi_handle_member<T>
constexpr auto handle(T const& t) {
    return t.mpi_handle();
}

template <builtin_handle T>
constexpr T handle(T t) noexcept {
    return t;
}

// ──────────────────────────────────────────────────────────────────────────────
// native_handle_ptr() dispatch — priority: native_handle_traits > member > builtin passthrough
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
    requires detail::handle_traits_has_handle_ptr<std::remove_cvref_t<T>>
constexpr auto handle_ptr(T& t) {
    return handle_traits<std::remove_cvref_t<T>>::handle_ptr(t);
}

template <typename T>
    requires(!detail::handle_traits_has_handle_ptr<std::remove_cvref_t<T>>) && detail::has_mpi_handle_ptr_member<T>
constexpr auto handle_ptr(T& t) {
    return t.mpi_handle_ptr();
}

template <builtin_handle T>
constexpr T* handle_ptr(T& t) noexcept {
    return &t;
}

template <builtin_handle T>
constexpr T* handle_ptr(T* t) noexcept {
    return t;
}

// ──────────────────────────────────────────────────────────────────────────────
// handle_range() dispatch — priority: handle_range_traits > mpi_handle_range() > contiguous range passthrough
//
// Tiers 1 and 2 return decltype(auto) so that a view returned by value (e.g. std::span) stays a value and a
// container returned by reference stays a reference. The passthrough tier only accepts lvalues and returns them by
// reference. MPI writes completed handles (MPI_REQUEST_NULL) back into the range, so it must never be a copy.
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
    requires detail::handle_range_traits_has_handle_range<std::remove_cvref_t<T>>
constexpr decltype(auto) handle_range(T&& t) {
    return handle_range_traits<std::remove_cvref_t<T>>::handle_range(t);
}

template <typename T>
    requires(!detail::handle_range_traits_has_handle_range<std::remove_cvref_t<T>>)
            && detail::has_mpi_handle_range_member<T>
constexpr decltype(auto) handle_range(T&& t) {
    return t.mpi_handle_range();
}

template <typename T>
    requires(!detail::handle_range_traits_has_handle_range<std::remove_cvref_t<T>>)
            && (!detail::has_mpi_handle_range_member<T>) && contiguous_handle_range<T>
constexpr T& handle_range(T& t) noexcept {
    return t;
}

template <typename T, typename HandleType>
concept convertible_to_mpi_handle = builtin_handle<HandleType> && requires(T const& t) {
    { handle(t) } -> std::same_as<HandleType>;
};

template <typename T, typename HandleType>
concept convertible_to_mpi_handle_ptr = builtin_handle<HandleType> && requires(T& t) {
    { handle_ptr(t) } -> std::same_as<HandleType*>;
};

/// A type whose handle_range() is a writable, contiguous range of `HandleType` that refers to the caller's storage:
/// either an lvalue reference or a borrowed view (never a container returned by value, since MPI writes into it).
template <typename T, typename HandleType>
concept convertible_to_mpi_handle_range = builtin_handle<HandleType> && requires(T& t) {
    { handle_range(t) } -> contiguous_handle_range_of<HandleType>;
};

// ──────────────────────────────────────────────────────────────────────────────
// to_rank() / to_tag() — customization points for MPI rank and tag values.
//
// Dispatch priority:
//   1. rank_traits<T>::rank(t) / tag_traits<T>::tag(t)  — for types you don't own
//   2. Scoped enums — via std::to_underlying, then narrowing to int
//   3. Implicit int conversion — int, unscoped enums, etc.
//
// To support a typed rank wrapper or a scoped enum with non-int underlying type:
//
//   template <>
//   struct rank_traits<Rank> {
//       static int rank(Rank const& r) { return r.value; }
//   };
//
//   enum class Tag : uint8_t { DATA = 0, SYNC = 1 };
//   // No specialization needed — to_underlying handles it.
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
struct rank_traits {};

template <typename T>
struct tag_traits {};

namespace detail {

template <typename T>
concept rank_traits_has_rank = requires(T const& t) {
    { rank_traits<T>::rank(t) } -> std::same_as<int>;
};

template <typename T>
concept tag_traits_has_tag = requires(T const& t) {
    { tag_traits<T>::tag(t) } -> std::same_as<int>;
};
} // namespace detail

template <typename T>
    requires detail::rank_traits_has_rank<std::remove_cvref_t<T>>
constexpr int to_rank(T const& t) {
    return rank_traits<std::remove_cvref_t<T>>::rank(t);
}

template <typename T>
    requires(!detail::rank_traits_has_rank<std::remove_cvref_t<T>>) && is_scoped_enum_v<std::remove_cvref_t<T>>
constexpr int to_rank(T t) {
    return static_cast<int>(static_cast<std::underlying_type_t<std::remove_cvref_t<T>>>(t));
}

template <typename T>
    requires(!detail::rank_traits_has_rank<std::remove_cvref_t<T>>) && (!is_scoped_enum_v<std::remove_cvref_t<T>>)
            && std::convertible_to<T, int>
constexpr int to_rank(T t) {
    return static_cast<int>(t);
}

template <typename T>
    requires detail::tag_traits_has_tag<std::remove_cvref_t<T>>
constexpr int to_tag(T const& t) {
    return tag_traits<std::remove_cvref_t<T>>::tag(t);
}

template <typename T>
    requires(!detail::tag_traits_has_tag<std::remove_cvref_t<T>>) && is_scoped_enum_v<std::remove_cvref_t<T>>
constexpr int to_tag(T t) {
    return static_cast<int>(static_cast<std::underlying_type_t<std::remove_cvref_t<T>>>(t));
}

template <typename T>
    requires(!detail::tag_traits_has_tag<std::remove_cvref_t<T>>) && (!is_scoped_enum_v<std::remove_cvref_t<T>>)
            && std::convertible_to<T, int>
constexpr int to_tag(T t) {
    return static_cast<int>(t);
}

template <typename T>
concept rank = requires(T const& t) {
    { to_rank(t) } -> std::same_as<int>;
};

template <typename T>
concept tag = requires(T const& t) {
    { to_tag(t) } -> std::same_as<int>;
};

} // namespace mpi::experimental
