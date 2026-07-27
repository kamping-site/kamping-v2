// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

/// @file
/// @brief A registry for committed MPI datatypes with pipe-able view adaptors.

#include <optional>
#include <typeindex>
#include <unordered_map>

#include <mpi.h>

#include "kamping/types/mpi_type_traits.hpp"
#include "kamping/types/scoped_datatype.hpp"
#include "kamping/v2/kassert.hpp"
#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/payload.hpp"
#include "kamping/v2/views/with_type_view.hpp"
#include "kamping/v2/views/with_value_type_view.hpp"

namespace kamping::v2 {

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

    /// @brief Registers a caller-supplied `MPI_Datatype` for `T`, without requiring a
    /// `kamping::types::mpi_type_traits<T>` specialization.
    ///
    /// `T` is used purely as the lookup key (`std::type_index(typeid(T))`) — e.g. a byte-blob
    /// type for a struct that has no trait specialization written for it. The pool wraps `dt`
    /// in a `types::ScopedDatatype`, which commits it (a no-op if already committed) and frees
    /// it on pool destruction; the pool owns the handle from this point on, the caller must not
    /// free it separately.
    ///
    /// Idempotent-keep-existing: if `T` is already registered (by this overload or the
    /// trait-based one above), the existing handle is kept. Callers are expected to be able to
    /// call this repeatedly for the same `T` (e.g. once per `request_reply()` invocation) without
    /// leaking — a helper like `byte_serialized<T>::data_type()` builds a fresh derived datatype
    /// on every call, so on a redundant registration `dt` is freed immediately (`MPI_Type_free`,
    /// legal on both committed and uncommitted derived types) rather than silently discarded.
    ///
    /// @pre `dt` must not be a predefined/named handle (e.g. `MPI_INT`) — freeing it (either via
    /// `ScopedDatatype` on first registration, or directly here on a redundant one) would be
    /// invalid. This overload is for derived (contiguous/struct) types only.
    /// @tparam T The lookup key; no trait requirement.
    /// @param dt The (possibly uncommitted) `MPI_Datatype` to register for `T`.
    /// @return The committed `MPI_Datatype` now owned by the pool for `T`.
    template <typename T>
    MPI_Datatype register_type(MPI_Datatype dt) {
        std::type_index idx = std::type_index(typeid(T));
        auto            it  = _types.find(idx);
        if (it == _types.end()) {
            it = _types.emplace(idx, dt).first;
        } else if (dt != MPI_DATATYPE_NULL) {
            MPI_Type_free(&dt);
        }
        return it->second.data_type();
    }

    /// @brief Looks up the `MPI_Datatype` for `T` without registering it.
    ///
    /// For builtin types, always returns the predefined MPI type. Otherwise (including types
    /// with no `mpi_type_traits<T>` specialization — e.g. registered only via the one-shot
    /// `register_type<T>(MPI_Datatype)` overload), looks `T` up in the pool by key and returns
    /// `std::nullopt` if it has not been registered.
    /// @tparam T The type to look up; no trait requirement.
    /// @return The `MPI_Datatype` for `T`, or `std::nullopt` if not registered.
    template <typename T>
    std::optional<MPI_Datatype> find() const {
        // Nested (not combined-&&) on purpose: mpi_type_traits<T>::has_to_be_committed is a hard
        // error for a genuinely trait-less T (the empty primary template has no such member), and
        // a single `if constexpr (has_static_type_v<T> && !mpi_type_traits<T>::has_to_be_committed)`
        // does NOT get concept-style short-circuit treatment — both operands must be well-formed
        // regardless of the left side's value. Nesting ensures the inner check is only instantiated
        // once has_static_type_v<T> is already known true.
        if constexpr (kamping::types::has_static_type_v<T>) {
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
    ///
    /// Uses `mpi_element_type_t`, not `payload_element_t`: `R` is always taken to be an already-flat
    /// buffer here (this is the plain buffer-type channel, `mpi::experimental::type()`), never
    /// reinterpreted as a still-structured (nested/sparse/value_destination_pair) source — see
    /// mpi_element_type_t's doc comment in payload.hpp for why that distinction matters.
    template <typename R>
        requires requires { typename kamping::v2::mpi_element_type_t<std::remove_cvref_t<R>>; }
                 && kamping::types::has_static_type_v<kamping::v2::mpi_element_type_t<std::remove_cvref_t<R>>>
    auto operator()(R&& r, type_pool const& pool) {
        using elem_t = kamping::v2::mpi_element_type_t<std::remove_cvref_t<R>>;
        auto dt      = pool.find<elem_t>();
        KAMPING_V2_ASSERT(dt.has_value(), "Type not registered in pool; call register_type<T>() first.");
        return kamping::v2::with_type_view(std::forward<R>(r), *dt);
    }

    /// Overload accepting any type satisfying @ref has_pool (e.g. `comm_view_with_pool`).
    template <typename R, typename Env>
        requires has_pool<std::remove_cvref_t<Env>>
                 && requires { typename kamping::v2::mpi_element_type_t<std::remove_cvref_t<R>>; }
                 && kamping::types::has_static_type_v<kamping::v2::mpi_element_type_t<std::remove_cvref_t<R>>>
    auto operator()(R&& r, Env&& env) {
        return (*this)(std::forward<R>(r), env.pool());
    }
};

struct with_auto_pool_fn {
    /// Attaches the MPI datatype for the buffer's element type, registering it in the pool if needed.
    /// Works with standard ranges and any buffer type exposing a public `value_type` member.
    ///
    /// Uses `mpi_element_type_t`, not `payload_element_t` — see @ref with_pool_fn.
    template <typename R>
        requires requires { typename kamping::v2::mpi_element_type_t<std::remove_cvref_t<R>>; }
                 && kamping::types::has_static_type_v<kamping::v2::mpi_element_type_t<std::remove_cvref_t<R>>>
    auto operator()(R&& r, type_pool& pool) {
        using elem_t = kamping::v2::mpi_element_type_t<std::remove_cvref_t<R>>;
        return kamping::v2::with_type_view(std::forward<R>(r), pool.register_type<elem_t>());
    }

    /// Overload accepting any type satisfying @ref has_pool (e.g. `comm_view_with_pool`).
    template <typename R, typename Env>
        requires has_pool<std::remove_cvref_t<Env>>
                 && requires { typename kamping::v2::mpi_element_type_t<std::remove_cvref_t<R>>; }
                 && kamping::types::has_static_type_v<kamping::v2::mpi_element_type_t<std::remove_cvref_t<R>>>
    auto operator()(R&& r, Env&& env) {
        return (*this)(std::forward<R>(r), env.pool());
    }
};

struct with_value_pool_fn {
    /// Attaches the MPI datatype for the buffer's *payload* (value-slot) type from a
    /// pre-populated pool — the value-type channel counterpart of @ref with_pool_fn. The
    /// payload is resolved via @ref kamping::v2::payload_element_t, so it differs from the
    /// buffer's own element type for structured shapes (value_destination_pair,
    /// sparse/nested send buffers). Asserts that the payload type has been registered; call
    /// `pool.register_type<T>()` (or the one-shot `register_type<T>(MPI_Datatype)` overload)
    /// beforehand.
    template <typename R>
        requires requires { typename kamping::v2::payload_element_t<std::remove_cvref_t<R>>; }
    auto operator()(R&& r, type_pool const& pool) {
        using payload_t = kamping::v2::payload_element_t<std::remove_cvref_t<R>>;
        auto dt         = pool.find<payload_t>();
        KAMPING_V2_ASSERT(dt.has_value(), "Payload type not registered in pool; call register_type<T>() first.");
        return kamping::v2::with_value_type_view(std::forward<R>(r), *dt);
    }

    /// Overload accepting any type satisfying @ref has_pool (e.g. `comm_view_with_pool`).
    template <typename R, typename Env>
        requires has_pool<std::remove_cvref_t<Env>>
                 && requires { typename kamping::v2::payload_element_t<std::remove_cvref_t<R>>; }
    auto operator()(R&& r, Env&& env) {
        return (*this)(std::forward<R>(r), env.pool());
    }
};

struct with_auto_value_pool_fn {
    /// Attaches the MPI datatype for the buffer's payload type, registering it in the pool if
    /// needed — the value-type channel counterpart of @ref with_auto_pool_fn. Only covers
    /// payload types with a `kamping::types::mpi_type_traits<T>` specialization (mirroring
    /// `with_auto_pool`'s constraint); a payload type with no trait must be pre-registered via
    /// the one-shot `register_type<T>(MPI_Datatype)` overload and attached via @ref
    /// with_value_pool_fn instead.
    template <typename R>
        requires requires { typename kamping::v2::payload_element_t<std::remove_cvref_t<R>>; }
                 && kamping::types::has_static_type_v<kamping::v2::payload_element_t<std::remove_cvref_t<R>>>
    auto operator()(R&& r, type_pool& pool) {
        using payload_t = kamping::v2::payload_element_t<std::remove_cvref_t<R>>;
        return kamping::v2::with_value_type_view(std::forward<R>(r), pool.register_type<payload_t>());
    }

    /// Overload accepting any type satisfying @ref has_pool (e.g. `comm_view_with_pool`).
    template <typename R, typename Env>
        requires has_pool<std::remove_cvref_t<Env>>
                 && requires { typename kamping::v2::payload_element_t<std::remove_cvref_t<R>>; }
                 && kamping::types::has_static_type_v<kamping::v2::payload_element_t<std::remove_cvref_t<R>>>
    auto operator()(R&& r, Env&& env) {
        return (*this)(std::forward<R>(r), env.pool());
    }
};
} // namespace detail

/// @brief Pipe adaptor that attaches an MPI datatype from a pre-populated @ref type_pool.
///
/// The element type is resolved via @ref kamping::v2::payload_element_t: standard ranges use
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
/// The element type is resolved via @ref kamping::v2::payload_element_t: standard ranges use
/// `range_value_t`; custom buffer wrappers (e.g. kokkos_view, sycl_view) are supported via
/// a public `value_type` member.  Unlike @ref with_pool, the type need not be registered in
/// advance — the pool commits it the first time this adaptor is applied.
///
/// @code
/// kamping::v2::bcast(v | kamping::v2::views::with_auto_pool(pool));
/// kamping::v2::send(kokkos_buf | kamping::v2::views::with_auto_pool(pool), 1, comm);
/// @endcode
inline constexpr kamping::v2::adaptor<1, detail::with_auto_pool_fn> with_auto_pool{};

/// @brief Pipe adaptor that attaches an MPI datatype to the value-type channel
/// (`kamping::v2::value_type()`, NOT `mpi::experimental::type()`) from a pre-populated
/// @ref type_pool — the payload-channel counterpart of @ref with_pool.
///
/// The payload type is resolved via @ref kamping::v2::payload_element_t: for structured send
/// buffers (value_destination_pair, sparse/nested) this is the payload/value slot, distinct
/// from the buffer's own range element type; for a plain range it is the same as with_pool.
/// The type must have been registered via `pool.register_type<T>()` (trait-based) or
/// `pool.register_type<T>(dt)` (one-shot, no trait required) before piping; an assertion fires
/// at runtime otherwise. Use @ref with_auto_value_pool to register lazily instead.
///
/// @code
/// type_pool p;
/// dstl::request_reply(pairs | kamping::v2::views::with_value_pool(p), ...);
/// @endcode
inline constexpr kamping::v2::adaptor<1, detail::with_value_pool_fn> with_value_pool{};

/// @brief Pipe adaptor that attaches an MPI datatype to the value-type channel from a
/// @ref type_pool, registering on first use — the payload-channel counterpart of
/// @ref with_auto_pool. Only covers payload types with a `mpi_type_traits<T>` specialization;
/// see @ref with_auto_value_pool_fn.
///
/// @code
/// type_pool p;
/// dstl::request_reply(pairs | kamping::v2::views::with_auto_value_pool(p), ...);
/// @endcode
inline constexpr kamping::v2::adaptor<1, detail::with_auto_value_pool_fn> with_auto_value_pool{};
} // namespace views
} // namespace kamping::v2
