// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

/// @file
/// @brief Attribute caching on MPI objects: kind tags selecting the `MPI_*_attr` functions, and a typed keyval whose
/// attributes own their values. See also `mpi/cache.hpp` for a cache that owns its values itself, and
/// `docs/attribute_caching.md` for when to use which.

#include <concepts>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

/// @brief Kinds of MPI objects that support attribute caching.
///
/// The handle type alone cannot select the attribute API: in some implementations (e.g. MPICH) `MPI_Comm`, `MPI_Win`
/// and `MPI_Datatype` are all `int`. Each tag names its handle type and the matching `MPI_*_attr` functions.
namespace attribute {

/// @brief Attributes on communicators (`MPI_Comm_*_attr`).
struct comm {
    using handle_type     = MPI_Comm;
    using delete_function = MPI_Comm_delete_attr_function;

    static int create_keyval(delete_function* del, int* keyval, void* extra_state) {
        return MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, del, keyval, extra_state);
    }
    static int free_keyval(int* keyval) {
        return MPI_Comm_free_keyval(keyval);
    }
    static int set_attr(handle_type h, int keyval, void* value) {
        return MPI_Comm_set_attr(h, keyval, value);
    }
    static int get_attr(handle_type h, int keyval, void* value, int* found) {
        return MPI_Comm_get_attr(h, keyval, value, found);
    }
    static int delete_attr(handle_type h, int keyval) {
        return MPI_Comm_delete_attr(h, keyval);
    }
};

/// @brief Attributes on windows (`MPI_Win_*_attr`).
struct win {
    using handle_type     = MPI_Win;
    using delete_function = MPI_Win_delete_attr_function;

    static int create_keyval(delete_function* del, int* keyval, void* extra_state) {
        return MPI_Win_create_keyval(MPI_WIN_NULL_COPY_FN, del, keyval, extra_state);
    }
    static int free_keyval(int* keyval) {
        return MPI_Win_free_keyval(keyval);
    }
    static int set_attr(handle_type h, int keyval, void* value) {
        return MPI_Win_set_attr(h, keyval, value);
    }
    static int get_attr(handle_type h, int keyval, void* value, int* found) {
        return MPI_Win_get_attr(h, keyval, value, found);
    }
    static int delete_attr(handle_type h, int keyval) {
        return MPI_Win_delete_attr(h, keyval);
    }
};

/// @brief Attributes on datatypes (`MPI_Type_*_attr`).
struct type {
    using handle_type     = MPI_Datatype;
    using delete_function = MPI_Type_delete_attr_function;

    static int create_keyval(delete_function* del, int* keyval, void* extra_state) {
        return MPI_Type_create_keyval(MPI_TYPE_NULL_COPY_FN, del, keyval, extra_state);
    }
    static int free_keyval(int* keyval) {
        return MPI_Type_free_keyval(keyval);
    }
    static int set_attr(handle_type h, int keyval, void* value) {
        return MPI_Type_set_attr(h, keyval, value);
    }
    static int get_attr(handle_type h, int keyval, void* value, int* found) {
        return MPI_Type_get_attr(h, keyval, value, found);
    }
    static int delete_attr(handle_type h, int keyval) {
        return MPI_Type_delete_attr(h, keyval);
    }
};

} // namespace attribute

/// @brief A kind tag from @ref attribute: names a handle type and its attribute functions.
template <typename Kind>
concept attribute_kind =
    builtin_handle<typename Kind::handle_type>
    && requires(typename Kind::handle_type h, typename Kind::delete_function* del, int* keyval, void* ptr, int* found) {
           { Kind::create_keyval(del, keyval, ptr) } -> std::same_as<int>;
           { Kind::free_keyval(keyval) } -> std::same_as<int>;
           { Kind::set_attr(h, 0, ptr) } -> std::same_as<int>;
           { Kind::get_attr(h, 0, ptr, found) } -> std::same_as<int>;
           { Kind::delete_attr(h, 0) } -> std::same_as<int>;
       };

/// @brief A typed keyval: caches one `T` per MPI object, owned by the object's attribute.
///
/// Each value lives as long as its attribute: it is destroyed when its object is freed (`MPI_Comm_free`,
/// `MPI_Win_free`, `MPI_Type_free`) or when it is removed with @ref erase. Destroying the keyval only frees the key:
/// values already attached stay alive until their objects are freed, but can no longer be looked up. Values on objects
/// that are never freed (e.g. `MPI_COMM_WORLD`) are therefore never destroyed.
///
/// This suits stateless interfaces, where nothing but the MPI object is available to hold the cached state (e.g.
/// behind a C API). When an explicit owner is available, prefer `cache<Kind, T>`, which also destroys its values when
/// it is destroyed.
///
/// **Collective use.** If creating or destroying a `T` is collective (e.g. a window on a communicator), then
/// @ref get_or_create and @ref erase are collective too, and freeing the object runs the destructor of its value.
///
/// **Lifetime of the key.** A keyval must be destroyed while MPI is usable. For a keyval in static storage, which is
/// destroyed after `MPI_Finalize`, the destructor skips freeing the key once MPI has been finalized (World model). In
/// the Sessions model this cannot be detected; destroy the keyval before the last session is finalized.
///
/// The delete callback runs inside MPI, so `T`'s destructor must not throw. Duplicated objects
/// (`MPI_Comm_dup`, `MPI_Type_dup`) do not inherit cached values.
///
/// @tparam Kind The kind of MPI object, one of the tags in @ref attribute.
/// @tparam T    The cached value type. Need not be movable: it is constructed in place from the factory's result.
template <attribute_kind Kind, typename T>
    requires std::is_nothrow_destructible_v<T>
class keyval {
public:
    /// The raw MPI handle type of the objects carrying values.
    using handle_type = typename Kind::handle_type;

    /// @brief Creates a new key.
    /// @throws mpi_error if the key cannot be created.
    keyval() {
        int err = Kind::create_keyval(&delete_fn, &_keyval, nullptr);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    }

    keyval(keyval const&)            = delete;
    keyval& operator=(keyval const&) = delete;

    /// @brief Takes over `other`'s key; `other` is left without one.
    keyval(keyval&& other) noexcept : _keyval(std::exchange(other._keyval, MPI_KEYVAL_INVALID)) {}

    /// @brief Frees this key, then takes over `other`'s.
    keyval& operator=(keyval&& other) noexcept {
        if (this != &other) {
            free_if_valid();
            _keyval = std::exchange(other._keyval, MPI_KEYVAL_INVALID);
        }
        return *this;
    }

    /// @brief Frees the key; attached values stay alive until their objects are freed (see class documentation).
    ~keyval() {
        free_if_valid();
    }

    /// @brief Returns the value attached to `obj`, creating it with `make()` if there is none.
    ///
    /// @param obj  The MPI object, as a raw handle or any wrapper convertible to it.
    /// @param make Called only on a miss; must return a `T` by value, which is constructed in place.
    /// @return The attached value; stays valid until `obj` is freed or the value is erased.
    /// @throws mpi_error if an attribute call fails; nothing is attached in that case.
    template <convertible_to_mpi_handle<handle_type> Obj, std::invocable F>
        requires std::same_as<std::invoke_result_t<F>, T>
    T& get_or_create(Obj const& obj, F&& make) {
        if (T* existing = find(obj)) {
            return *existing;
        }
        auto value = std::unique_ptr<T>(new T(std::invoke(std::forward<F>(make))));
        if (int err = Kind::set_attr(handle(obj), _keyval, value.get()); err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        return *value.release(); // now owned by the attribute; delete_fn destroys it
    }

    /// @return The value attached to `obj`, or `nullptr` if there is none.
    /// @throws mpi_error if the attribute lookup fails.
    template <convertible_to_mpi_handle<handle_type> Obj>
    [[nodiscard]] T* find(Obj const& obj) const {
        void* value = nullptr;
        int   found = 0;
        if (int err = Kind::get_attr(handle(obj), _keyval, &value, &found); err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        return found ? static_cast<T*>(value) : nullptr;
    }

    /// @brief Destroys the value attached to `obj`, if any.
    /// @return `true` if a value was destroyed.
    /// @throws mpi_error if an attribute call fails.
    template <convertible_to_mpi_handle<handle_type> Obj>
    bool erase(Obj const& obj) {
        if (find(obj) == nullptr) {
            return false;
        }
        if (int err = Kind::delete_attr(handle(obj), _keyval); err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        return true;
    }

private:
    // Called by MPI when an object with an attached value is freed or its attribute is deleted.
    static int delete_fn(handle_type /* h */, int /* keyval */, void* value, void* /* extra_state */) noexcept {
        delete static_cast<T*>(value);
        return MPI_SUCCESS;
    }

    void free_if_valid() noexcept {
        if (_keyval == MPI_KEYVAL_INVALID) {
            return;
        }
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            Kind::free_keyval(&_keyval);
        }
        _keyval = MPI_KEYVAL_INVALID;
    }

    int _keyval = MPI_KEYVAL_INVALID;
};

} // namespace mpi::experimental
