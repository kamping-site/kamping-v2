// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

/// @file
/// @brief An owning cache of per-object values attached via MPI attributes. See `docs/attribute_caching.md` for
/// when to use it instead of `keyval` (`mpi/attribute.hpp`).

#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include "mpi/attribute.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

/// @brief Owning cache of one `T` per MPI object, attached to the object as an attribute.
///
/// The cache owns its values; each object carries a non-owning attribute pointing to its value, which makes lookup
/// O(1). A value is destroyed when
/// - its object is freed (`MPI_Comm_free`, `MPI_Win_free`, `MPI_Type_free`), via the attribute's delete callback,
/// - it is removed with @ref erase, or
/// - the cache is destroyed, which removes all remaining values in the order they were created.
///
/// Each cache has its own keyval, so independent caches on the same object do not interfere, and no global state is
/// involved: the cache works the same with the World and the Sessions model. Where no explicit owner is available
/// (stateless interfaces, e.g. behind a C API), use `keyval<Kind, T>`, whose values are owned by the objects instead.
///
/// **Collective use.** If creating or destroying a `T` is collective (e.g. a window on a communicator), then
/// @ref get_or_create, @ref erase and the destructor are collective too: all processes must call them for the same
/// objects in the same order. Destroying the cache in creation order makes its cleanup consistent across processes.
/// The cache must be destroyed before MPI is finalized.
///
/// The delete callback runs inside MPI, so `T`'s destructor must not throw. Duplicated objects
/// (`MPI_Comm_dup`, `MPI_Type_dup`) do not inherit cached values.
///
/// @tparam Kind The kind of MPI object, one of the tags in @ref attribute.
/// @tparam T    The cached value type. Need not be movable: it is constructed in place from the factory's result.
template <attribute_kind Kind, typename T>
    requires std::is_nothrow_destructible_v<T>
class cache {
public:
    /// The raw MPI handle type of the cached objects.
    using handle_type = typename Kind::handle_type;

    /// @brief Creates an empty cache with its own keyval.
    /// @throws mpi_error if the keyval cannot be created.
    cache() : _impl(std::make_unique<impl>()) {
        int err = Kind::create_keyval(&delete_fn, &_impl->keyval, _impl.get());
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    }

    cache(cache const&)            = delete;
    cache& operator=(cache const&) = delete;

    /// @brief Takes over `other`'s values. A moved-from cache may only be destroyed or assigned to.
    cache(cache&& other) noexcept = default;

    /// @brief Destroys this cache's values, then takes over `other`'s.
    cache& operator=(cache&& other) noexcept {
        if (this != &other) {
            clear_and_free_keyval();
            _impl = std::move(other._impl);
        }
        return *this;
    }

    /// @brief Destroys all remaining values in creation order and frees the keyval (see class documentation).
    ~cache() {
        clear_and_free_keyval();
    }

    /// @brief Returns the value cached for `obj`, creating it with `make()` if there is none.
    ///
    /// @param obj  The MPI object, as a raw handle or any wrapper convertible to it.
    /// @param make Called only on a miss; must return a `T` by value, which is constructed in place.
    /// @return The cached value; stays valid until it is destroyed (see class documentation).
    /// @throws mpi_error if an attribute call fails; nothing is cached in that case.
    template <convertible_to_mpi_handle<handle_type> Obj, std::invocable F>
        requires std::same_as<std::invoke_result_t<F>, T>
    T& get_or_create(Obj const& obj, F&& make) {
        if (T* existing = find(obj)) {
            return *existing;
        }
        handle_type const h     = handle(obj);
        auto              value = std::unique_ptr<T>(new T(std::invoke(std::forward<F>(make))));
        _impl->entries.emplace_back(h, std::move(value));
        T* const ptr = _impl->entries.back().second.get();
        int      err = Kind::set_attr(h, _impl->keyval, ptr);
        if (err != MPI_SUCCESS) {
            _impl->entries.pop_back();
            throw mpi_error(err);
        }
        return *ptr;
    }

    /// @return The value cached for `obj`, or `nullptr` if there is none.
    /// @throws mpi_error if the attribute lookup fails.
    template <convertible_to_mpi_handle<handle_type> Obj>
    [[nodiscard]] T* find(Obj const& obj) {
        return lookup(handle(obj));
    }

    /// @return The value cached for `obj`, or `nullptr` if there is none. The cache owns its values, so a const cache
    ///         gives read-only access to them.
    /// @throws mpi_error if the attribute lookup fails.
    template <convertible_to_mpi_handle<handle_type> Obj>
    [[nodiscard]] T const* find(Obj const& obj) const {
        return lookup(handle(obj));
    }

    /// @brief Destroys the value cached for `obj`, if any.
    /// @return `true` if a value was destroyed.
    /// @throws mpi_error if an attribute call fails.
    template <convertible_to_mpi_handle<handle_type> Obj>
    bool erase(Obj const& obj) {
        if (find(obj) == nullptr) {
            return false;
        }
        int err = Kind::delete_attr(handle(obj), _impl->keyval);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        return true;
    }

    /// @return The number of cached values.
    [[nodiscard]] std::size_t size() const noexcept {
        return _impl->entries.size();
    }

    /// @return `true` if no values are cached.
    [[nodiscard]] bool empty() const noexcept {
        return _impl->entries.empty();
    }

private:
    // Shared by both find overloads, which only differ in the constness of the result.
    T* lookup(handle_type h) const {
        void* value = nullptr;
        int   found = 0;
        int   err   = Kind::get_attr(h, _impl->keyval, &value, &found);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        return found ? static_cast<T*>(value) : nullptr;
    }

    // Heap-allocated so that its address, passed to MPI as the keyval's extra_state, survives moves of the cache.
    struct impl {
        int keyval = MPI_KEYVAL_INVALID;
        // In creation order. unique_ptr keeps each value's address stable while the vector grows, since the
        // attributes point at the values.
        std::vector<std::pair<handle_type, std::unique_ptr<T>>> entries;
    };

    // Called by MPI when an object with a cached value is freed or its attribute is deleted.
    static int delete_fn(handle_type h, int /* keyval */, void* /* value */, void* extra_state) noexcept {
        std::erase_if(static_cast<impl*>(extra_state)->entries, [h](auto const& entry) { return entry.first == h; });
        return MPI_SUCCESS;
    }

    void clear_and_free_keyval() noexcept {
        if (!_impl) {
            return; // moved from
        }
        while (!_impl->entries.empty()) {
            // On success, delete_fn removes the entry; on failure, drop it here so the loop terminates.
            if (Kind::delete_attr(_impl->entries.front().first, _impl->keyval) != MPI_SUCCESS) {
                _impl->entries.erase(_impl->entries.begin());
            }
        }
        Kind::free_keyval(&_impl->keyval);
        _impl.reset();
    }

    std::unique_ptr<impl> _impl;
};

} // namespace mpi::experimental
