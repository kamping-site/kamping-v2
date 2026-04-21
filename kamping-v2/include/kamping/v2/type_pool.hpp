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
#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/with_type_view.hpp"

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

namespace views {
namespace detail {
struct with_pool_fn {
    /// Attaches the MPI datatype for the range's value type from a pre-populated pool.
    /// Asserts that the type has been registered; call `pool.register_type<T>()` beforehand.
    template <typename R>
        requires std::ranges::range<R> && kamping::types::has_static_type_v<std::ranges::range_value_t<R>>
    auto operator()(R&& r, type_pool const& pool) {
        auto dt = pool.find<std::ranges::range_value_t<R>>();
        KAMPING_ASSERT(dt.has_value(), "Type not registered in pool; call register_type<T>() first.");
        return kamping::v2::with_type_view(std::forward<R>(r), *dt);
    }
};

struct with_auto_pool_fn {
    /// Attaches the MPI datatype for the range's value type, registering it in the pool if needed.
    template <typename R>
        requires std::ranges::range<R> && kamping::types::has_static_type_v<std::ranges::range_value_t<R>>
    auto operator()(R&& r, type_pool& pool) {
        return kamping::v2::with_type_view(
            std::forward<R>(r),
            pool.register_type<std::ranges::range_value_t<R>>()
        );
    }
};
} // namespace detail

/// @brief Pipe adaptor that attaches an MPI datatype from a pre-populated @ref type_pool.
///
/// The type must have been registered via `pool.register_type<T>()` before piping; an assertion
/// fires at runtime otherwise. Use @ref with_auto_pool to register lazily instead.
///
/// @code
/// pool.register_type<MyStruct>();
/// kamping::v2::bcast(v | kamping::v2::views::with_pool(pool));
/// @endcode
inline constexpr kamping::v2::adaptor<1, detail::with_pool_fn> with_pool{};

/// @brief Pipe adaptor that attaches an MPI datatype from a @ref type_pool, registering on first use.
///
/// Unlike @ref with_pool, the type need not be registered in advance — the pool commits it
/// the first time this adaptor is applied. Use @ref with_pool when you want explicit control
/// over when types are committed.
///
/// @code
/// kamping::v2::bcast(v | kamping::v2::views::with_auto_pool(pool));
/// @endcode
inline constexpr kamping::v2::adaptor<1, detail::with_auto_pool_fn> with_auto_pool{};
} // namespace views
} // namespace kamping::v2
