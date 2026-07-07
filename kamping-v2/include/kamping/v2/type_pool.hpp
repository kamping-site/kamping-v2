// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

/// @file
/// @brief A registry for committed MPI datatypes with pipe-able view adaptors.

#include <optional>
#include <ranges>
#include <typeindex>
#include <unordered_map>

#include <mpi.h>

#include "kamping/types/mpi_type_traits.hpp"
#include "kamping/types/scoped_datatype.hpp"
#include "kamping/v2/kassert.hpp"
#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/with_type_view.hpp"

namespace kamping::v2 {

namespace detail {

/// @brief Resolves the element type of a buffer for MPI type dispatch.
///
/// Priority:
///   1. `std::ranges::range_value_t<R>` — standard ranges (std::vector, std::span, …)
///   2. `R::value_type` member — custom buffer wrappers that expose a public value type
///      (e.g. kokkos_view, sycl_view)
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

template <typename R>
using mpi_element_type_t = typename mpi_element_type<R>::type;

} // namespace detail

/// @brief Registry that owns committed MPI datatypes for the lifetime of the pool.
///
/// Builtin types (e.g. `int`, `double`) are always available without registration.
/// Derived types (e.g. contiguous or struct types) must be registered via @ref register_type
/// before use; the pool commits and owns the resulting `MPI_Datatype`.
///
/// The pool is move-only: copying is disabled because `MPI_Datatype` handles are not duplicable.
class type_pool {
private:
    std::unordered_map<std::type_index, types::ScopedDatatype> _types;

public:
    type_pool()                            = default;
    type_pool(type_pool const&)            = delete;
    type_pool& operator=(type_pool const&) = delete;
    type_pool(type_pool&&)                 = default;
    type_pool& operator=(type_pool&&)      = default;

    /// @brief Registers `T` with the pool and returns its `MPI_Datatype`.
    ///
    /// For builtin types, returns the predefined MPI type immediately without storing anything.
    /// For derived types, commits the type on the first call and returns the cached handle on
    /// subsequent calls (idempotent).
    /// @tparam T A type satisfying `has_static_type_v`.
    /// @return The committed `MPI_Datatype` for `T`.
    template <typename T>
        requires kamping::types::has_static_type_v<T>
    MPI_Datatype register_type() {
        if constexpr (!kamping::types::mpi_type_traits<T>::has_to_be_committed) {
            return kamping::types::mpi_type_traits<T>::data_type();
        } else {
            std::type_index idx = std::type_index(typeid(T));
            auto            it  = _types.find(idx);
            if (it == _types.end()) {
                it = _types.emplace(idx, kamping::types::mpi_type_traits<T>::data_type()).first;
            }
            return it->second.data_type();
        }
    }

    /// @brief Looks up the `MPI_Datatype` for `T` without registering it.
    ///
    /// For builtin types, always returns the predefined MPI type.
    /// For derived types, returns `std::nullopt` if `register_type<T>()` has not been called.
    /// @tparam T A type satisfying `has_static_type_v`.
    /// @return The `MPI_Datatype` for `T`, or `std::nullopt` if not registered.
    template <typename T>
        requires kamping::types::has_static_type_v<T>
    std::optional<MPI_Datatype> find() const {
        if constexpr (!kamping::types::mpi_type_traits<T>::has_to_be_committed) {
            return kamping::types::mpi_type_traits<T>::data_type();
        } else {
            std::type_index idx = std::type_index(typeid(T));
            auto            it  = _types.find(idx);
            if (it == _types.end()) {
                return std::nullopt;
            }
            return it->second.data_type();
        }
    }
};

/// @brief Concept matching any type that exposes a `pool()` method returning a `type_pool&`.
///
/// Satisfied by @ref comm_view_with_pool, enabling `views::with_auto_pool(env)` and
/// `views::with_pool(env)` to accept a pooled communicator directly.
template <typename T>
concept has_pool = requires(T& t) {
    { t.pool() } -> std::same_as<type_pool&>;
};

namespace views {
namespace detail {
struct with_pool_fn {
    /// Attaches the MPI datatype for the buffer's element type from a pre-populated pool.
    /// Works with standard ranges and any buffer type exposing a public `value_type` member.
    /// Asserts that the type has been registered; call `pool.register_type<T>()` beforehand.
    template <typename R>
        requires requires { typename kamping::v2::detail::mpi_element_type_t<std::remove_cvref_t<R>>; }
                 && kamping::types::has_static_type_v<kamping::v2::detail::mpi_element_type_t<std::remove_cvref_t<R>>>
    auto operator()(R&& r, type_pool const& pool) {
        using elem_t = kamping::v2::detail::mpi_element_type_t<std::remove_cvref_t<R>>;
        auto dt      = pool.find<elem_t>();
        KAMPING_V2_ASSERT(dt.has_value(), "Type not registered in pool; call register_type<T>() first.");
        return kamping::v2::with_type_view(std::forward<R>(r), *dt);
    }

    /// Overload accepting any type satisfying @ref has_pool (e.g. `comm_view_with_pool`).
    template <typename R, typename Env>
        requires has_pool<std::remove_cvref_t<Env>>
                 && requires { typename kamping::v2::detail::mpi_element_type_t<std::remove_cvref_t<R>>; }
                 && kamping::types::has_static_type_v<kamping::v2::detail::mpi_element_type_t<std::remove_cvref_t<R>>>
    auto operator()(R&& r, Env&& env) {
        return (*this)(std::forward<R>(r), env.pool());
    }
};

struct with_auto_pool_fn {
    /// Attaches the MPI datatype for the buffer's element type, registering it in the pool if needed.
    /// Works with standard ranges and any buffer type exposing a public `value_type` member.
    template <typename R>
        requires requires { typename kamping::v2::detail::mpi_element_type_t<std::remove_cvref_t<R>>; }
                 && kamping::types::has_static_type_v<kamping::v2::detail::mpi_element_type_t<std::remove_cvref_t<R>>>
    auto operator()(R&& r, type_pool& pool) {
        using elem_t = kamping::v2::detail::mpi_element_type_t<std::remove_cvref_t<R>>;
        return kamping::v2::with_type_view(std::forward<R>(r), pool.register_type<elem_t>());
    }

    /// Overload accepting any type satisfying @ref has_pool (e.g. `comm_view_with_pool`).
    template <typename R, typename Env>
        requires has_pool<std::remove_cvref_t<Env>>
                 && requires { typename kamping::v2::detail::mpi_element_type_t<std::remove_cvref_t<R>>; }
                 && kamping::types::has_static_type_v<kamping::v2::detail::mpi_element_type_t<std::remove_cvref_t<R>>>
    auto operator()(R&& r, Env&& env) {
        return (*this)(std::forward<R>(r), env.pool());
    }
};
} // namespace detail

/// @brief Pipe adaptor that attaches an MPI datatype from a pre-populated @ref type_pool.
///
/// The element type is resolved via @ref kamping::v2::detail::mpi_element_type_t: standard ranges use
/// `range_value_t`; custom buffer wrappers (e.g. kokkos_view, sycl_view) are supported via
/// a public `value_type` member.  The type must have been registered via
/// `pool.register_type<T>()` before piping; an assertion fires at runtime otherwise.
/// Use @ref with_auto_pool to register lazily instead.
///
/// @code
/// pool.register_type<MyStruct>();
/// kamping::v2::bcast(v | kamping::v2::views::with_pool(pool));
/// kamping::v2::send(kokkos_buf | kamping::v2::views::with_pool(pool), 1, comm);
/// @endcode
inline constexpr kamping::v2::adaptor<1, detail::with_pool_fn> with_pool{};

/// @brief Pipe adaptor that attaches an MPI datatype from a @ref type_pool, registering on first use.
///
/// The element type is resolved via @ref kamping::v2::detail::mpi_element_type_t: standard ranges use
/// `range_value_t`; custom buffer wrappers (e.g. kokkos_view, sycl_view) are supported via
/// a public `value_type` member.  Unlike @ref with_pool, the type need not be registered in
/// advance — the pool commits it the first time this adaptor is applied.
///
/// @code
/// kamping::v2::bcast(v | kamping::v2::views::with_auto_pool(pool));
/// kamping::v2::send(kokkos_buf | kamping::v2::views::with_auto_pool(pool), 1, comm);
/// @endcode
inline constexpr kamping::v2::adaptor<1, detail::with_auto_pool_fn> with_auto_pool{};
} // namespace views
} // namespace kamping::v2
