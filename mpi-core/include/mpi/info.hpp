// This file is part of KaMPIng.
//
// Copyright 2022-2026 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

/// @file
/// @brief Owning (`info`) and non-owning (`info_view`) wrappers for `MPI_Info`.

#pragma once

#include <charconv>
#include <concepts>
#include <iterator>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/thread_level.hpp"

namespace mpi::experimental {

// ─── info_value_traits ───────────────────────────────────────────────────────
//
// Extensible traits for type-safe `set<T>()` / `get<T>()`.
//
// The primary template is intentionally left undefined — specialize it for
// your own types.  Built-in specializations cover:
//   - std::string      (identity)
//   - bool             ("true" / "false"; "1" / "0" also accepted on read)
//   - integral types   (std::to_string / std::from_chars)
//
// A specialization must provide:
//   @code
//   static std::string to_string(T const& v);   // T → string for info::set<T>
//   static T from_string(std::string const& s); // string → T for info::get<T>
//                                               // may throw on bad input
//   @endcode

template <typename T>
struct info_value_traits; ///< Primary template — undefined; specialize for custom types.

/// @brief Specialization for `std::string` (identity conversion).
template <>
struct info_value_traits<std::string> {
    static std::string to_string(std::string const& v) {
        return v;
    }
    static std::string from_string(std::string const& s) {
        return s;
    }
};

/// @brief Specialization for `bool` (`"true"` / `"false"`).
template <>
struct info_value_traits<bool> {
    static std::string to_string(bool v) {
        return v ? "true" : "false";
    }
    static bool from_string(std::string const& s) {
        if (s == "true" || s == "1") return true;
        if (s == "false" || s == "0") return false;
        throw std::invalid_argument("info_value_traits<bool>::from_string: cannot parse \"" + s + "\"");
    }
};

/// @brief Specialization for non-bool integral types (via `std::to_string` / `std::from_chars`).
template <typename T>
    requires std::integral<T> && (!std::same_as<T, bool>)
struct info_value_traits<T> {
    static std::string to_string(T v) {
        return std::to_string(v);
    }
    static T from_string(std::string const& s) {
        T    result{};
        auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), result);
        if (ec != std::errc{}) {
            throw std::invalid_argument("info_value_traits::from_string: cannot parse \"" + s + "\"");
        }
        return result;
    }
};

/// @brief Specialization for `ThreadLevel` — stored as its underlying integer value.
template <>
struct info_value_traits<ThreadLevel> {
    static std::string to_string(ThreadLevel v) {
        return std::to_string(static_cast<int>(v));
    }
    static ThreadLevel from_string(std::string const& s) {
        return static_cast<ThreadLevel>(info_value_traits<int>::from_string(s));
    }
};

// ─── entry_iterator / entry_sentinel ─────────────────────────────────────────
//
// Input iterator over (key, value) pairs of an MPI_Info object, using the
// C++20 sentinel pattern.
//
//  entry_iterator — stores MPI_Info + current index; operator* queries MPI.
//  entry_sentinel — stores nkeys; compared against the iterator via operator==.
//
// This separates the "where am I?" (iterator) from "when am I done?" (sentinel),
// keeping the iterator lean (no _nkeys field) and operator== a single comparison.

namespace detail {

// Fetch the nth key-value pair from an MPI_Info object.
inline std::pair<std::string, std::string> fetch_info_entry(MPI_Info info, int idx) {
    char key_buf[MPI_MAX_INFO_KEY + 1] = {};
    MPI_Info_get_nthkey(info, idx, key_buf);
    std::string key(key_buf); // std::string(char*) stops at first \0

    int flag = 0;
    int vlen = 0;
#if MPI_VERSION >= 4
    int buflen = 0;
    MPI_Info_get_string(info, key_buf, &buflen, nullptr, &flag);
    vlen = buflen - 1; // buflen includes the null terminator
#else
    MPI_Info_get_valuelen(info, key_buf, &vlen, &flag);
#endif
    std::string value;
    if (flag && vlen > 0) {
        value.resize(static_cast<std::size_t>(vlen));
#if MPI_VERSION >= 4
        int buflen2 = vlen + 1;
        MPI_Info_get_string(info, key_buf, &buflen2, value.data(), &flag);
#else
        MPI_Info_get(info, key_buf, vlen, value.data(), &flag);
#endif
        // Truncate at the first embedded \0 (defensive)
        if (auto pos = value.find('\0'); pos != std::string::npos) {
            value.resize(pos);
        }
    }
    return {std::move(key), std::move(value)};
}

} // namespace detail

// Forward declarations — entry_iterator befriends the free operator==,
// whose definition follows after entry_sentinel is complete.
class  entry_iterator;
struct entry_sentinel;
[[nodiscard]] bool operator==(entry_iterator const& it, entry_sentinel s) noexcept;

/// @brief Input iterator over (key, value) pairs of an `MPI_Info` object.
///
/// Designed as a C++20 sentinel iterator: the end position is represented by
/// @ref entry_sentinel, not by a second `entry_iterator`.  Use via
/// `info_accessors::begin()` / `info_accessors::end()`, or the
/// `info_accessors::entries()` range directly.
class entry_iterator {
public:
    using value_type        = std::pair<std::string, std::string>;
    using difference_type   = std::ptrdiff_t;
    using reference         = value_type; ///< Returned by value; no persistent MPI address.
    using pointer           = void;
    using iterator_category = std::input_iterator_tag;

    entry_iterator(MPI_Info info, int idx) noexcept : _info(info), _idx(idx) {}

    /// @brief Fetch the current (key, value) pair from MPI.
    [[nodiscard]] value_type operator*() const {
        return detail::fetch_info_entry(_info, _idx);
    }

    entry_iterator& operator++() noexcept {
        ++_idx;
        return *this;
    }
    entry_iterator operator++(int) noexcept {
        auto tmp = *this;
        ++_idx;
        return tmp;
    }

    [[nodiscard]] bool operator==(entry_iterator const& o) const noexcept {
        return _info == o._info && _idx == o._idx;
    }

private:
    // Grant the free operator== access to _idx.
    friend bool operator==(entry_iterator const& it, entry_sentinel s) noexcept;

    MPI_Info _info;
    int      _idx;
};

/// @brief Sentinel for `entry_iterator` — holds the total number of keys.
struct entry_sentinel {
    int nkeys;
};

/// @brief The iterator is exhausted when its index reaches the total key count.
[[nodiscard]] inline bool operator==(entry_iterator const& it, entry_sentinel s) noexcept {
    return it._idx >= s.nkeys;
}

static_assert(std::sentinel_for<entry_sentinel, entry_iterator>);

// ─── info_accessors CRTP mixin ────────────────────────────────────────────────
//
// Shared accessors for `info` (owning) and `info_view` (non-owning).
// Derived must implement:
//   `MPI_Info mpi_handle() const noexcept;`

template <typename Derived>
class info_accessors {
    [[nodiscard]] MPI_Info h() const noexcept {
        return static_cast<Derived const*>(this)->mpi_handle();
    }

    /// @brief Returns the value length (without null terminator) for @p key,
    ///        or `std::nullopt` if absent.
    [[nodiscard]] std::optional<int> value_length(char const* key) const {
        int flag = 0;
        int vlen = 0;
#if MPI_VERSION >= 4
        // MPI-4: buflen includes the null terminator on output.
        int buflen = 0;
        MPI_Info_get_string(h(), key, &buflen, nullptr, &flag);
        vlen = buflen - 1;
#else
        MPI_Info_get_valuelen(h(), key, &vlen, &flag);
#endif
        if (!flag) return std::nullopt;
        return vlen;
    }

public:
    // ── count ────────────────────────────────────────────────────────────────

    /// @return Number of key–value pairs.
    [[nodiscard]] int nkeys() const {
        int  n   = 0;
        auto err = MPI_Info_get_nkeys(h(), &n);
        if (err != MPI_SUCCESS) throw mpi_error(err);
        return n;
    }

    // ── key lookup ───────────────────────────────────────────────────────────

    /// @brief Return the @p n-th key (0-indexed), null-truncated at the first `\0`.
    [[nodiscard]] std::string nth_key(int n) const {
        char buf[MPI_MAX_INFO_KEY + 1] = {};
        auto err                       = MPI_Info_get_nthkey(h(), n, buf);
        if (err != MPI_SUCCESS) throw mpi_error(err);
        return std::string(buf); // stops at first \0
    }

    // ── get ──────────────────────────────────────────────────────────────────

    /// @brief Look up @p key; returns `std::nullopt` if the key is absent.
    [[nodiscard]] std::optional<std::string> get(std::string_view key) const {
        std::string const key_s(key);
        auto const        vlen = value_length(key_s.c_str());
        if (!vlen) return std::nullopt;

        std::string value(static_cast<std::size_t>(*vlen), '\0');
        int         flag = 0;
#if MPI_VERSION >= 4
        int buflen = *vlen + 1;
        auto err   = MPI_Info_get_string(h(), key_s.c_str(), &buflen, value.data(), &flag);
        if (err != MPI_SUCCESS) throw mpi_error(err);
#else
        auto err = MPI_Info_get(h(), key_s.c_str(), *vlen, value.data(), &flag);
        if (err != MPI_SUCCESS) throw mpi_error(err);
#endif
        // Truncate at the first embedded \0 (defensive)
        if (auto pos = value.find('\0'); pos != std::string::npos) {
            value.resize(pos);
        }
        return value;
    }

    /// @brief Type-safe get.  Requires `info_value_traits<T>` to be specialized.
    template <typename T>
    [[nodiscard]] std::optional<T> get(std::string_view key) const {
        auto maybe = get(key);
        if (!maybe) return std::nullopt;
        return info_value_traits<T>::from_string(*maybe);
    }

    // ── contains ─────────────────────────────────────────────────────────────

    /// @return `true` if @p key is present.
    [[nodiscard]] bool contains(std::string_view key) const {
        std::string const key_s(key);
        return value_length(key_s.c_str()).has_value();
    }

    // ── set ──────────────────────────────────────────────────────────────────

    /// @brief Set @p key to @p value (creates or overwrites).
    void set(std::string_view key, std::string_view value) {
        std::string const key_s(key);
        std::string const val_s(value);
        auto              err = MPI_Info_set(h(), key_s.c_str(), val_s.c_str());
        if (err != MPI_SUCCESS) throw mpi_error(err);
    }

    /// @brief Type-safe set.  Requires `info_value_traits<T>` to be specialized.
    ///
    /// Note: this overload is excluded when `T` is already implicitly convertible
    /// to `std::string_view` — the non-template overload above is preferred then.
    template <typename T>
        requires(!std::convertible_to<T, std::string_view>)
    void set(std::string_view key, T const& value) {
        std::string const str = info_value_traits<T>::to_string(value);
        set(key, std::string_view(str));
    }

    // ── erase ────────────────────────────────────────────────────────────────

    /// @brief Remove @p key.
    /// @throws mpi_error if the MPI call fails (including when the key is absent).
    void erase(std::string_view key) {
        std::string const key_s(key);
        auto              err = MPI_Info_delete(h(), key_s.c_str());
        if (err != MPI_SUCCESS) throw mpi_error(err);
    }

    // ── iteration ────────────────────────────────────────────────────────────

    /// @brief Iterator to the first key–value pair.
    [[nodiscard]] entry_iterator begin() const {
        return entry_iterator(h(), 0);
    }

    /// @brief Sentinel marking the end of iteration.
    [[nodiscard]] entry_sentinel end() const {
        return entry_sentinel{nkeys()};
    }

    /// @brief Lazy range of all (key, value) pairs.
    ///
    /// Returns a `std::ranges::subrange` that can be passed directly to range
    /// algorithms and supports structured bindings in range-for:
    /// @code
    ///   for (auto [key, val] : my_info.entries()) { ... }
    ///   auto n = std::ranges::distance(my_info.entries());
    /// @endcode
    [[nodiscard]] auto entries() const {
        return std::ranges::subrange(begin(), end());
    }

    // ── escape hatch ─────────────────────────────────────────────────────────

    /// @return The underlying `MPI_Info` handle.
    [[nodiscard]] MPI_Info native() const noexcept { return h(); }
};

// ─── info_view ───────────────────────────────────────────────────────────────

/// @brief Non-owning view over an existing `MPI_Info`.
///
/// Does not free the handle on destruction.  Use for `MPI_INFO_NULL`, handles
/// borrowed from the MPI implementation, or anywhere lifetimes are managed externally.
///
/// Satisfies `convertible_to_mpi_handle<MPI_Info>` so it can be passed directly
/// to any `mpi::experimental::` operation that accepts an info argument.
class info_view : public info_accessors<info_view> {
public:
    /// @brief Wrap @p info without taking ownership.  The handle must outlive this view.
    explicit info_view(MPI_Info info) noexcept : _info(info) {}

    /// @return The underlying `MPI_Info` (for `mpi::experimental::handle()` dispatch).
    [[nodiscard]] MPI_Info mpi_handle() const noexcept { return _info; }

private:
    MPI_Info _info;
};

// ─── info ─────────────────────────────────────────────────────────────────────

/// @brief Owning RAII wrapper for `MPI_Info`.
///
/// Calls `MPI_Info_create` on construction and `MPI_Info_free` on destruction.
/// Move-only: if you need an independent copy, call `dup()` explicitly.
///
/// Satisfies `convertible_to_mpi_handle<MPI_Info>` so it can be passed directly
/// to any `mpi::experimental::` operation that accepts an info argument.
class info : public info_accessors<info> {
public:
    /// @brief Create a new empty info object via `MPI_Info_create`.
    /// @throws mpi_error if the MPI call fails.
    info() {
        auto err = MPI_Info_create(&_info);
        if (err != MPI_SUCCESS) throw mpi_error(err);
    }

    info(info const&)            = delete;
    info& operator=(info const&) = delete;

    /// @brief Move constructor — transfers ownership; the moved-from object holds `MPI_INFO_NULL`.
    info(info&& other) noexcept : _info(std::exchange(other._info, MPI_INFO_NULL)) {}

    /// @brief Move assignment — frees the current handle then transfers ownership.
    info& operator=(info&& other) noexcept {
        if (this != &other) {
            free_if_valid();
            _info = std::exchange(other._info, MPI_INFO_NULL);
        }
        return *this;
    }

    /// @brief Free the info object (unless moved from).
    ~info() noexcept {
        free_if_valid();
    }

    /// @brief Duplicate this info object into a new, independently owned copy.
    /// @return A new `info` with the same key–value pairs.
    /// @throws mpi_error if `MPI_Info_dup` fails.
    [[nodiscard]] info dup() const {
        MPI_Info copy = MPI_INFO_NULL;
        auto     err  = MPI_Info_dup(_info, &copy);
        if (err != MPI_SUCCESS) throw mpi_error(err);
        return info(copy, adopt_t{});
    }

    /// @return The underlying `MPI_Info` (for `mpi::experimental::handle()` dispatch).
    [[nodiscard]] MPI_Info mpi_handle() const noexcept { return _info; }

private:
    struct adopt_t {};

    /// @brief Adopt an already-created handle (used by `dup()`).
    info(MPI_Info h, adopt_t) noexcept : _info(h) {}

    void free_if_valid() noexcept {
        if (_info != MPI_INFO_NULL) {
            MPI_Info_free(&_info);
            _info = MPI_INFO_NULL;
        }
    }

    MPI_Info _info = MPI_INFO_NULL;
};

} // namespace mpi::experimental
