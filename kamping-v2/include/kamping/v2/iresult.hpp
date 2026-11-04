// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include "kamping/v2/result.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "mpi/handle.hpp"
#include "mpi/request.hpp"

namespace kamping::v2 {

// Lifetime invariant shared by both specialisations: an owned buffer must outlive the operation using it.
//
// An iresult must be completed with wait() or a successful test() before it is destroyed or move-assigned over;
// otherwise the `_request` member calls std::terminate() — see mpi::experimental::request. The class is [[nodiscard]],
// so discarding the iresult returned by a non-blocking call (which would drop it immediately) is a compile warning.
//
//   - Destruction: the `_request` member is declared *after* the buffer, so it is destroyed first: it terminates
//     *before* the buffer of a still-running operation is released, never after.
//   - Move assignment: moves into a temporary and swaps, so the previously held request and buffer are dropped by
//     the temporary's destructor in that same order. (A defaulted move assignment would release the old buffer
//     before checking the old request.)
//
// For the same reason the request cannot live in a common base class: base subobjects are destroyed after members.

// Primary template — specialised for one or two buffer types below.
template <typename... Bufs>
class [[nodiscard]] iresult;

/// The pending result of a non-blocking operation with one buffer, e.g. `isend` or `irecv`.
///
/// A buffer passed as an rvalue is moved into the `iresult`, which owns it until the operation has completed:
/// it is only accessible through `wait()` or a successful `test()`, so it cannot be touched while MPI may still
/// use it. A buffer passed as an lvalue is borrowed; you must not access it before completion.
///
/// @code
/// auto pending = kamping::v2::irecv(std::vector<int>{} | kamping::v2::views::resize, source, comm);
/// // ... other work ...
/// std::vector<int> data = pending.wait().underlying();
/// @endcode
///
/// An `iresult` is move-only, and moving it keeps the buffer in place. It must be completed with `wait()` or a
/// successful `test()`: destroying or assigning over an `iresult` whose operation is still active prints a
/// diagnostic and calls `std::terminate()`, instead of releasing a buffer that MPI may still be using.
template <typename Buf>
class [[nodiscard]] iresult<Buf> {
    using view_t = kamping::v2::all_t<Buf>;

    std::unique_ptr<view_t> _view;
    // Declared after _view so that it is destroyed first (see the lifetime invariant above).
    mpi::experimental::request _request;

    // Returns the buffer with correct ownership semantics:
    //   - lvalue Buf (T&):           ref_view<T> stored → base() → T&
    //   - non-view rvalue Buf (T):   owning_view<T> stored → move(*_view).base() → T&&
    //   - view rvalue Buf (View):    view stored as-is → move(*_view) → View&&
    decltype(auto) extract_buf() {
        if constexpr (std::is_lvalue_reference_v<Buf>) {
            return _view->base();
        } else if constexpr (std::derived_from<std::remove_cvref_t<Buf>, kamping::v2::view_interface_base>) {
            return std::move(*_view);
        } else {
            return std::move(*_view).base();
        }
    }

public:
    explicit iresult(Buf&& buf) : _view(std::make_unique<view_t>(kamping::v2::all(std::forward<Buf>(buf)))) {}

    iresult(iresult&&) noexcept = default;

    /// Takes over `other`'s operation and buffer. Terminates if this iresult's operation is still active.
    iresult& operator=(iresult&& other) noexcept {
        iresult tmp(std::move(other));
        std::swap(_view, tmp._view);
        std::swap(_request, tmp._request);
        return *this; // tmp's destructor checks the old request, then releases the old buffer
    }

    /// @internal For the MPI call site to fill in the request (satisfies convertible_to_mpi_handle_ptr<MPI_Request>).
    MPI_Request* mpi_handle_ptr() noexcept {
        return _request.mpi_handle_ptr();
    }

    /// @internal The view of the buffer, to issue the MPI call that fills in the request.
    view_t& view() {
        return *_view;
    }

    /// @internal The view of the buffer.
    view_t const& view() const {
        return *_view;
    }

    // ── Completion ───────────────────────────────────────────────────────────

    /// Blocks until the operation completes, then returns the buffer: an owned buffer by value (the iresult is
    /// left empty), a borrowed buffer by reference.
    ///
    /// @param status Optional: filled with the status of the operation. Defaults to `MPI_STATUS_IGNORE`.
    template <mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    decltype(auto) wait(Status&& status = MPI_STATUS_IGNORE) {
        _request.wait(std::forward<Status>(status));
        return extract_buf();
    }

    /// Checks whether the operation has completed, without blocking.
    ///
    /// For a borrowed buffer, returns whether the operation has completed. For an owned buffer, returns a
    /// `std::optional` holding the buffer if it has (the iresult is then left empty), and `std::nullopt` if it is
    /// still active, in which case you can test again or `wait()`.
    ///
    /// @param status Optional: filled with the status of the operation if it has completed. Defaults to
    /// `MPI_STATUS_IGNORE`.
    template <mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    auto test(Status&& status = MPI_STATUS_IGNORE) {
        bool const done = _request.test(std::forward<Status>(status));
        if constexpr (kamping::v2::borrowed_buffer<view_t>) {
            return done;
        } else {
            using value_t = std::remove_reference_t<decltype(extract_buf())>;
            if (done)
                return std::optional<value_t>{extract_buf()};
            return std::optional<value_t>{std::nullopt};
        }
    }
};

/// The pending result of a non-blocking operation with a send and a receive buffer, e.g. `isendrecv`.
///
/// Like the one-buffer `iresult`: buffers passed as rvalues are owned until completion, buffers passed as lvalues
/// are borrowed, and the operation must be completed with `wait()` or a successful `test()`. Both return the
/// buffers as a `result`.
template <typename SBuf, typename RBuf>
class [[nodiscard]] iresult<SBuf, RBuf> {
    std::unique_ptr<result<SBuf, RBuf>> _result;
    // Declared after _result so that it is destroyed first (see the lifetime invariant above).
    mpi::experimental::request _request;

public:
    explicit iresult(SBuf&& sbuf, RBuf&& rbuf)
        : _result(
              std::make_unique<result<SBuf, RBuf>>(
                  result<SBuf, RBuf>{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)}
              )
          ) {}

    iresult(iresult&&) noexcept = default;

    /// Takes over `other`'s operation and buffers. Terminates if this iresult's operation is still active.
    iresult& operator=(iresult&& other) noexcept {
        iresult tmp(std::move(other));
        std::swap(_result, tmp._result);
        std::swap(_request, tmp._request);
        return *this; // tmp's destructor checks the old request, then releases the old buffers
    }

    /// @internal For the MPI call site to fill in the request (satisfies convertible_to_mpi_handle_ptr<MPI_Request>).
    MPI_Request* mpi_handle_ptr() noexcept {
        return _request.mpi_handle_ptr();
    }

    // ── Buffer access ─────────────────────────────────────────────────────────

    /// @internal The send buffer, to issue the MPI call that fills in the request.
    SBuf& send() {
        return _result->send;
    }
    /// @internal The receive buffer, to issue the MPI call that fills in the request.
    RBuf& recv() {
        return _result->recv;
    }

    // ── Completion ────────────────────────────────────────────────────────────

    /// Blocks until the operation completes, then returns both buffers: owned buffers by value (the iresult is left
    /// empty), borrowed buffers by reference.
    ///
    /// @param status Optional: filled with the status of the operation. Defaults to `MPI_STATUS_IGNORE`.
    template <mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    result<SBuf, RBuf> wait(Status&& status = MPI_STATUS_IGNORE) {
        _request.wait(std::forward<Status>(status));
        return std::move(*_result);
    }

    /// Checks whether the operation has completed, without blocking.
    ///
    /// @param status Optional: filled with the status of the operation if it has completed. Defaults to
    /// `MPI_STATUS_IGNORE`.
    /// @return Both buffers, as from `wait()`, if the operation has completed; `std::nullopt` if it is still active,
    /// in which case you can test again or `wait()`.
    template <mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    std::optional<result<SBuf, RBuf>> test(Status&& status = MPI_STATUS_IGNORE) {
        if (_request.test(std::forward<Status>(status)))
            return std::move(*_result);
        return std::nullopt;
    }
};

} // namespace kamping::v2
