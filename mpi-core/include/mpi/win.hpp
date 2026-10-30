// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

/// @file
/// @brief Owning (`win`) and non-owning (`win_view`) wrappers for `MPI_Win`.

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/group.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

// ── Free functions ───────────────────────────────────────────────────────────
// Two levels, as for communicators:
//   Level 1: mirrors the C API; writes into a caller-chosen output handle.
//   Level 2: RAII convenience; returns an owning `win` (defined after `win`).
// Synchronization is also available as a member via win_accessors.

/// @brief Collective: synchronize an active-target epoch (`MPI_Win_fence`).
///
/// @tparam Win Any type satisfying `convertible_to_mpi_handle<MPI_Win>`.
/// @param w         The window.
/// @param assertion Bitwise OR of `MPI_MODE_*` assertions (`MPI_MODE_NOPRECEDE`, `MPI_MODE_NOSUCCEED`, …).
/// @throws mpi_error if `MPI_Win_fence` fails.
template <convertible_to_mpi_handle<MPI_Win> Win = MPI_Win>
void fence(Win const& w, int assertion = 0) {
    int err = MPI_Win_fence(assertion, handle(w));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief Collective: allocate a window exposing `count` elements of `T` on each process (`MPI_Win_allocate`).
///
/// The displacement unit is `sizeof(T)`, so target displacements count elements of `T`. The memory is owned by MPI,
/// valid until the window is freed, and uninitialized.
///
/// @tparam Comm   Any type satisfying `convertible_to_mpi_handle<MPI_Comm>`.
/// @tparam NewWin Any type satisfying `convertible_to_mpi_handle_ptr<MPI_Win>`.
/// @tparam Info   Any type satisfying `convertible_to_mpi_handle<MPI_Info>`.
/// @param count   Number of elements of `T` exposed by this process (may differ between processes).
/// @param c       The communicator whose processes share the window.
/// @param base    Output: receives the address of this process's local memory.
/// @param new_win Output: receives the window handle.
/// @param info    Info hints (defaults to `MPI_INFO_NULL`).
/// @throws mpi_error if `MPI_Win_allocate` fails.
template <
    typename T,
    convertible_to_mpi_handle<MPI_Comm>    Comm   = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Win> NewWin = MPI_Win*,
    convertible_to_mpi_handle<MPI_Info>    Info   = MPI_Info>
    requires std::is_trivially_copyable_v<T>
void win_allocate(std::size_t count, Comm const& c, T*& base, NewWin& new_win, Info const& info = MPI_INFO_NULL) {
    int err = MPI_Win_allocate(
        static_cast<MPI_Aint>(count * sizeof(T)),
        static_cast<int>(sizeof(T)),
        handle(info),
        handle(c),
        &base,
        handle_ptr(new_win)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

// ── CRTP mixin ──────────────────────────────────────────────────────────────
// Provides accessors and synchronization for any window wrapper: `.group()`, `.fence()`.
// Derived must implement `mpi_handle() const → MPI_Win`.

template <typename Derived>
class win_accessors {
    [[nodiscard]] MPI_Win underlying() const noexcept {
        return static_cast<Derived const*>(this)->mpi_handle();
    }

public:
    /// @brief Extract the group of processes sharing this window.
    ///
    /// The returned `group` is owned by the caller.
    /// @throws mpi_error if `MPI_Win_get_group` fails.
    [[nodiscard]] mpi::experimental::group group() const {
        MPI_Group g   = MPI_GROUP_EMPTY;
        int       err = MPI_Win_get_group(underlying(), &g);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        return mpi::experimental::group::from_native(g);
    }

    /// @brief Collective: synchronize an active-target epoch (`MPI_Win_fence`).
    ///
    /// Delegates to the free function `mpi::experimental::fence`.
    /// @throws mpi_error if `MPI_Win_fence` fails.
    void fence(int assertion = 0) const {
        mpi::experimental::fence(underlying(), assertion);
    }
};

// ── win_view ─────────────────────────────────────────────────────────────────

/// @brief Non-owning wrapper around an `MPI_Win`.
///
/// Satisfies `convertible_to_mpi_handle<MPI_Win>`. Does not free the window on destruction — use the owning `win`
/// for that. Provides `.group()` / `.fence()` via `win_accessors`.
class win_view : public win_accessors<win_view> {
public:
    /// @brief Construct from a raw `MPI_Win`. The window must outlive this view.
    win_view(MPI_Win w) noexcept : _win(w) {}

    /// @return The underlying `MPI_Win` (for `handle()` dispatch).
    [[nodiscard]] MPI_Win mpi_handle() const noexcept {
        return _win;
    }

private:
    MPI_Win _win;
};

// ── win ──────────────────────────────────────────────────────────────────────

/// @brief Owning RAII wrapper for `MPI_Win`.
///
/// Move-only. Calls `MPI_Win_free` on destruction unless the stored handle is `MPI_WIN_NULL`. Freeing a window is
/// collective, so all processes of the window's group must destroy (or @ref free) it at the same point.
///
/// Satisfies `convertible_to_mpi_handle<MPI_Win>` and `convertible_to_mpi_handle_ptr<MPI_Win>`.
class win : public win_accessors<win> {
public:
    /// @brief Default-construct a null window (holds `MPI_WIN_NULL`).
    ///
    /// Useful as an output parameter for the Level 1 `win_allocate()`.
    win() noexcept = default;

    win(win const&)            = delete;
    win& operator=(win const&) = delete;

    /// @brief Move constructor — transfers ownership; moved-from holds `MPI_WIN_NULL`.
    win(win&& o) noexcept : _win(std::exchange(o._win, MPI_WIN_NULL)) {}

    /// @brief Move assignment — frees the current window then transfers ownership.
    win& operator=(win&& o) noexcept {
        if (this != &o) {
            free_if_valid();
            _win = std::exchange(o._win, MPI_WIN_NULL);
        }
        return *this;
    }

    /// @brief Free the window (unless moved from or null).
    ~win() noexcept {
        free_if_valid();
    }

    /// @brief Adopt an already-created `MPI_Win` (takes ownership).
    [[nodiscard]] static win from_native(MPI_Win w) noexcept {
        return win(w, adopt_t{});
    }

    /// @brief Explicitly free the window.
    ///
    /// After this call `*this` holds `MPI_WIN_NULL` — the destructor becomes a no-op. Use this instead of relying on
    /// the destructor when you need to handle errors from `MPI_Win_free`.
    ///
    /// The handle is exchanged to `MPI_WIN_NULL` before the MPI call so the destructor cannot double-free even if
    /// this throws.
    ///
    /// @throws mpi_error if `MPI_Win_free` fails.
    void free() {
        if (_win != MPI_WIN_NULL) {
            MPI_Win tmp = std::exchange(_win, MPI_WIN_NULL);
            int     err = MPI_Win_free(&tmp);
            if (err != MPI_SUCCESS) {
                throw mpi_error(err);
            }
        }
    }

    /// @brief Relinquish ownership; returns the raw `MPI_Win` handle.
    ///
    /// Leaves `*this` as `MPI_WIN_NULL`. The caller is responsible for calling `MPI_Win_free` on the returned handle.
    [[nodiscard]] MPI_Win release() noexcept {
        return std::exchange(_win, MPI_WIN_NULL);
    }

    /// @brief Implicit conversion to a non-owning view.
    ///
    /// The view borrows; it does not extend lifetime.
    operator win_view() const noexcept {
        return win_view{_win};
    }

    /// @return The underlying `MPI_Win` (for `handle()` dispatch).
    [[nodiscard]] MPI_Win mpi_handle() const noexcept {
        return _win;
    }

    /// @brief Pointer to the underlying `MPI_Win`, for use as an MPI out-parameter.
    ///
    /// **Precondition**: the window must be null (`MPI_WIN_NULL`). Writing into a non-null `win` bypasses
    /// `MPI_Win_free` and leaks the existing window.
    MPI_Win* mpi_handle_ptr() noexcept {
        return &_win;
    }

private:
    struct adopt_t {};

    explicit win(MPI_Win w, adopt_t) noexcept : _win(w) {}

    void free_if_valid() noexcept {
        if (_win != MPI_WIN_NULL) {
            MPI_Win_free(&_win);
            _win = MPI_WIN_NULL;
        }
    }

    MPI_Win _win = MPI_WIN_NULL;
};

static_assert(convertible_to_mpi_handle<win_view, MPI_Win>);
static_assert(convertible_to_mpi_handle<win, MPI_Win>);
static_assert(convertible_to_mpi_handle_ptr<win, MPI_Win>);

/// @brief Collective: allocate a window exposing `count` elements of `T` on each process, returning an owning `win`.
///
/// See the Level 1 overload for the semantics.
///
/// @tparam Comm Any type satisfying `convertible_to_mpi_handle<MPI_Comm>`.
/// @tparam Info Any type satisfying `convertible_to_mpi_handle<MPI_Info>`.
/// @return The window and this process's local, uninitialized part of it.
/// @throws mpi_error if `MPI_Win_allocate` fails.
template <
    typename T,
    convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm,
    convertible_to_mpi_handle<MPI_Info> Info = MPI_Info>
    requires std::is_trivially_copyable_v<T>
[[nodiscard]] std::pair<win, std::span<T>>
win_allocate(std::size_t count, Comm const& c, Info const& info = MPI_INFO_NULL) {
    T*  base = nullptr;
    win w;
    win_allocate(count, c, base, w, info);
    return {std::move(w), std::span<T>(base, count)};
}

} // namespace mpi::experimental
