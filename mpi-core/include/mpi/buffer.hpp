#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>

#include <mpi.h>

#include <kamping/types/builtin_types.hpp>

/// @file
/// MPI buffer contract: customization point (buffer_traits), accessor dispatch
/// (count / ptr / type / counts / displs), and buffer taxonomy concepts
/// (send_buffer, recv_buffer, data_buffer_v, …).
///
/// Three-tier accessor dispatch priority:
///   1. buffer_traits<T> specialization   — non-intrusive, for types you don't own
///   2. mpi_count() / mpi_ptr() / …       — intrusive member functions
///   3. std::ranges::size / ptr / …      — fallback for standard ranges
///
/// To adapt a third-party type non-intrusively, specialize buffer_traits<T>:
///
///   template <>
///   struct buffer_traits<MyType> {
///       static std::ptrdiff_t size(MyType const& t) { return t.n; }
///       static int const*     ptr(MyType const& t) { return t.ptr; } // send
///       static int*           ptr(MyType&       t) { return t.ptr; } // recv
///       static MPI_Datatype   type(MyType const& t) { return MPI_INT; }
///   };

namespace mpi::experimental {

// ──────────────────────────────────────────────────────────────────────────────
// Public customization point.
// ──────────────────────────────────────────────────────────────────────────────

/// Trait class for non-intrusive MPI buffer customization.
/// Only implement the members you need; the remaining accessors fall back to
/// the mpi_count() / mpi_ptr() / mpi_type() member functions.
template <typename T>
struct buffer_traits {};

/// A contiguous range of `int` — the element type used for MPI counts and displacements.
template <typename T>
concept count_range =
    std::ranges::contiguous_range<T> && std::same_as<int, std::remove_cvref_t<std::ranges::range_value_t<T>>>;

// ──────────────────────────────────────────────────────────────────────────────
// detail — implementation helpers for the accessor dispatch, not public API.
// ──────────────────────────────────────────────────────────────────────────────

namespace detail {

template <class T>
concept integer_like = std::integral<T> && !std::same_as<T, bool>;

template <class T>
concept ptr_to_object =
    std::is_pointer_v<T> && (std::is_object_v<std::remove_pointer_t<T>> || std::is_void_v<std::remove_pointer_t<T>>);

template <typename T>
concept range_of_builtin_mpi_type =
    std::ranges::range<T> && kamping::types::is_builtin_type_v<std::remove_cvref_t<std::ranges::range_value_t<T>>>;

// Tier-2: intrusive member function detection.

template <typename T>
concept has_count_member = requires(std::remove_reference_t<T> const& t) {
    { t.mpi_count() } -> integer_like;
};

template <typename T>
concept has_ptr_member = requires(T&& t) {
    { t.mpi_ptr() } -> ptr_to_object;
};

template <typename T>
concept has_type_member = requires(std::remove_reference_t<T> const& t) {
    { t.mpi_type() } -> std::convertible_to<MPI_Datatype>;
};

template <typename T>
concept has_counts_member = requires(std::remove_reference_t<T> const& t) {
    { t.mpi_counts() } -> count_range;
};

template <typename T>
concept has_displs_member = requires(std::remove_reference_t<T> const& t) {
    { t.mpi_displs() } -> count_range;
};

// Tier-1: buffer_traits specialization detection.
// buffer_traits is in the enclosing mpi::experimental namespace.

template <typename T>
concept traits_has_count = requires(T const& t) {
    { buffer_traits<T>::count(t) } -> integer_like;
};

template <typename T>
concept traits_has_const_ptr = requires(T const& t) {
    { buffer_traits<T>::ptr(t) } -> ptr_to_object;
};

template <typename T>
concept traits_has_ptr = requires(T& t) {
    { buffer_traits<T>::ptr(t) } -> ptr_to_object;
};

template <typename T>
concept traits_has_type = requires(T const& t) {
    { buffer_traits<T>::type(t) } -> std::convertible_to<MPI_Datatype>;
};

template <typename T>
concept traits_has_counts = requires(T const& t) {
    { buffer_traits<T>::counts(t) } -> count_range;
};

template <typename T>
concept traits_has_displs = requires(T const& t) {
    { buffer_traits<T>::displs(t) } -> count_range;
};

} // namespace detail

// ──────────────────────────────────────────────────────────────────────────────
// size() — priority: buffer_traits > mpi_count() > std::ranges::size
//
// Always call as count(x) (qualified) to suppress ADL and
// prevent ambiguity with std::size for standard containers.
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
constexpr auto count(T&&) = delete;

template <typename T>
    requires detail::traits_has_count<std::remove_cvref_t<T>>
constexpr auto count(T&& t) {
    return buffer_traits<std::remove_cvref_t<T>>::count(t);
}

template <typename T>
    requires(!detail::traits_has_count<std::remove_cvref_t<T>>) && detail::has_count_member<T>
constexpr auto count(T&& t) {
    return t.mpi_count();
}

template <typename T>
    requires(!detail::traits_has_count<std::remove_cvref_t<T>>) && (!detail::has_count_member<T>)
            && std::ranges::sized_range<T>
constexpr auto count(T&& t) {
    return std::ranges::size(std::forward<T>(t));
}

// ──────────────────────────────────────────────────────────────────────────────
// data() — priority: buffer_traits > mpi_data() > std::ranges::data
//
// std::forward preserves const-qualifiers so buffer_traits<T> can provide both
// a const overload (send) and a non-const overload (recv).
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
    requires(detail::traits_has_const_ptr<std::remove_cvref_t<T>> || detail::traits_has_ptr<std::remove_cvref_t<T>>)
constexpr auto ptr(T&& t) {
    return buffer_traits<std::remove_cvref_t<T>>::ptr(std::forward<T>(t));
}

template <typename T>
    requires(!detail::traits_has_const_ptr<std::remove_cvref_t<T>>)
            && (!detail::traits_has_ptr<std::remove_cvref_t<T>>) && detail::has_ptr_member<T>
constexpr auto ptr(T&& t) {
    return t.mpi_ptr();
}

template <typename T>
    requires(!detail::traits_has_const_ptr<std::remove_cvref_t<T>>)
            && (!detail::traits_has_ptr<std::remove_cvref_t<T>>) && (!detail::has_ptr_member<T>)
            && std::ranges::contiguous_range<T>
constexpr auto ptr(T&& t) {
    return std::ranges::data(std::forward<T>(t));
}

// ──────────────────────────────────────────────────────────────────────────────
// type() — priority: buffer_traits > mpi_type() > builtin type deduction
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
    requires detail::traits_has_type<std::remove_cvref_t<T>>
constexpr auto type(T&& t) {
    return buffer_traits<std::remove_cvref_t<T>>::type(t);
}

template <typename T>
    requires(!detail::traits_has_type<std::remove_cvref_t<T>>) && detail::has_type_member<T>
constexpr auto type(T&& t) {
    return t.mpi_type();
}

template <typename T>
    requires(!detail::traits_has_type<std::remove_cvref_t<T>>) && (!detail::has_type_member<T>)
            && detail::range_of_builtin_mpi_type<T>
constexpr auto type(T&& /* t */) {
    return kamping::types::builtin_type<std::remove_cvref_t<std::ranges::range_value_t<T>>>::data_type();
}

// ──────────────────────────────────────────────────────────────────────────────
// counts() / displs() — priority: buffer_traits > mpi_counts() / mpi_displs()
//
// counts() preserves const: non-const T → mutable span (int*), const T → read-only
// span (int const*). This allows infer() to write directly into the counts buffer
// of a non-const deferred buffer, and MPI wrappers to read it const afterwards.
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
    requires detail::traits_has_counts<std::remove_cvref_t<T>>
constexpr auto counts(T&& t) {
    return buffer_traits<std::remove_cvref_t<T>>::counts(std::forward<T>(t));
}

template <typename T>
    requires(!detail::traits_has_counts<std::remove_cvref_t<T>>) && detail::has_counts_member<T>
constexpr auto counts(T&& t) {
    return t.mpi_counts();
}

template <typename T>
    requires detail::traits_has_displs<std::remove_cvref_t<T>>
constexpr auto displs(T&& t) {
    return buffer_traits<std::remove_cvref_t<T>>::displs(t);
}

template <typename T>
    requires(!detail::traits_has_displs<std::remove_cvref_t<T>>) && detail::has_displs_member<T>
constexpr auto displs(T&& t) {
    return t.mpi_displs();
}

// ──────────────────────────────────────────────────────────────────────────────
// Buffer taxonomy concepts.
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
concept has_mpi_count = requires(T const& t) {
    { count(t) } -> detail::integer_like;
};

template <typename T>
concept has_mpi_ptr = requires(T&& t) {
    { ptr(t) } -> detail::ptr_to_object;
};

template <typename T>
concept has_mpi_type = requires(T const& t) {
    { type(t) } -> std::convertible_to<MPI_Datatype>;
};

template <typename T>
concept data_buffer = has_mpi_count<T> && has_mpi_ptr<T> && has_mpi_type<T>;

template <typename T>
concept send_buffer = data_buffer<T> && requires(T&& t) {
    { ptr(t) } -> std::convertible_to<void const*>;
};

template <typename T>
concept recv_buffer = data_buffer<T> && requires(T&& t) {
    { ptr(t) } -> std::convertible_to<void*>;
};

/// A buffer that can be used for both sending and receiving (i.e. satisfies recv_buffer).
template <typename T>
concept send_recv_buffer = recv_buffer<T>;

template <typename T>
concept has_mpi_counts = requires(T const& t) {
    { counts(t) } -> count_range;
};

/// counts() on a non-const object returns a mutable contiguous range of int —
/// used by infer() to write per-rank counts directly into the buffer.
template <typename T>
concept has_mpi_counts_mutable = requires(T& t) {
    { counts(t) } -> std::ranges::contiguous_range;
    requires std::same_as<int, std::remove_cvref_t<std::ranges::range_value_t<decltype(counts(t))>>>;
    { std::ranges::data(counts(t)) } -> std::convertible_to<int*>;
};

template <typename T>
concept has_mpi_displs = requires(T const& t) {
    { displs(t) } -> count_range;
};

/// Variadic buffer: data + type + per-rank counts + displacements.
/// Does NOT require a scalar mpi::count() — variadic MPI operations use the
/// per-rank counts array directly and never need the total element count.
template <typename T>
concept data_buffer_v = has_mpi_ptr<T> && has_mpi_type<T> && has_mpi_counts<T> && has_mpi_displs<T>;

template <typename T>
concept send_buffer_v = data_buffer_v<T> && requires(T&& t) {
    { ptr(t) } -> std::convertible_to<void const*>;
};

template <typename T>
concept recv_buffer_v = data_buffer_v<T> && requires(T&& t) {
    { ptr(t) } -> std::convertible_to<void*>;
};

} // namespace mpi::experimental
