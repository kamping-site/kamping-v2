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
// otherwise the `request` member calls std::terminate() — see mpi::experimental::request. The class is [[nodiscard]],
// so discarding the iresult returned by a non-blocking call (which would drop it immediately) is a compile warning.
//
//   - Destruction: the `request` member is declared *after* the buffer, so it is destroyed first: it terminates
//     *before* the buffer of a still-running operation is released, never after.
//   - Move assignment: moves into a temporary and swaps, so the previously held request and buffer are dropped by
//     the temporary's destructor in that same order. (A defaulted move assignment would release the old buffer
//     before checking the old request.)
//
// For the same reason the request cannot live in a common base class: base subobjects are destroyed after members.

// Primary template — specialised for one or two buffer types below.
template <typename... Bufs>
class [[nodiscard]] iresult;

/// Single-buffer non-blocking result (isend, irecv).
///
/// Stores the buffer on the heap via a unique_ptr<view_t> so that the data pointer
/// MPI captures during MPI_Isend/Irecv remains valid regardless of whether the
/// iresult handle itself is moved. The request is owned by a mpi::experimental::request member.
///
/// The view wrapping follows the usual all_t<Buf> convention:
///   - lvalue T&  → ref_view<T>:   non-owning; caller keeps the buffer alive.
///   - rvalue T   → owning_view<T>: owns the buffer until wait()/test() completes.
///   - view type  → stored as-is (all_t pass-through).
///
/// iresult is move-only: moving transfers ownership of the heap view (the view's
/// address is unchanged) and of the request. Destroying or overwriting an iresult whose operation
/// is still outstanding terminates the program (see the lifetime invariant above).
template <typename Buf>
class [[nodiscard]] iresult<Buf> {
    using view_t = kamping::v2::all_t<Buf>;

    std::unique_ptr<view_t> view_;
    // Declared after view_ so that it is destroyed first (see the lifetime invariant above).
    mpi::experimental::request request_;

    // Returns the buffer with correct ownership semantics:
    //   - lvalue Buf (T&):           ref_view<T> stored → base() → T&
    //   - non-view rvalue Buf (T):   owning_view<T> stored → move(*view_).base() → T&&
    //   - view rvalue Buf (View):    view stored as-is → move(*view_) → View&&
    decltype(auto) extract_buf() {
        if constexpr (std::is_lvalue_reference_v<Buf>) {
            return view_->base();
        } else if constexpr (std::derived_from<std::remove_cvref_t<Buf>, kamping::v2::view_interface_base>) {
            return std::move(*view_);
        } else {
            return std::move(*view_).base();
        }
    }

public:
    explicit iresult(Buf&& buf) : view_(std::make_unique<view_t>(kamping::v2::all(std::forward<Buf>(buf)))) {}

    iresult(iresult&&) noexcept = default;

    /// Releases this iresult's buffer and takes over `other`'s. Aborts if this iresult's operation is still
    /// outstanding.
    iresult& operator=(iresult&& other) noexcept {
        iresult tmp(std::move(other));
        std::swap(view_, tmp.view_);
        std::swap(request_, tmp.request_);
        return *this; // tmp's destructor checks the old request, then releases the old buffer
    }

    /// For the MPI call site to fill in the request (satisfies convertible_to_mpi_handle_ptr<MPI_Request>).
    MPI_Request* mpi_handle_ptr() noexcept {
        return request_.mpi_handle_ptr();
    }

    view_t& view() {
        return *view_;
    }

    view_t const& view() const {
        return *view_;
    }

    // ── Completion ───────────────────────────────────────────────────────────

    /// Blocks until the operation completes, then returns the buffer.
    ///   owned  (owning_view<T>): moves value out — iresult is left in moved-from state.
    ///   borrowed (ref_view<T>):  returns lvalue reference to the external buffer.
    ///   view pass-through:       moves the view out.
    template <mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    decltype(auto) wait(Status&& status = MPI_STATUS_IGNORE) {
        request_.wait(std::forward<Status>(status));
        return extract_buf();
    }

    /// Non-blocking completion check.
    ///   borrowed (ref_view<T> or borrowed view): returns bool.
    ///   owned (owning_view<T> or non-borrowed view): returns optional<T> / optional<View>.
    ///     Some on completion — buffer moved out, iresult spent.
    ///     nullopt on pending — iresult remains valid for retry or wait().
    template <mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    auto test(Status&& status = MPI_STATUS_IGNORE) {
        bool const done = request_.test(std::forward<Status>(status));
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

/// Two-buffer non-blocking result (non-blocking sendrecv, non-blocking collectives).
///
/// Stores a result<SBuf, RBuf> on the heap for pointer stability; the request is owned by a
/// mpi::experimental::request member. Mirrors iresult<Buf>:
///   - lvalue SBuf/RBuf: reference members — caller keeps the buffers alive.
///   - rvalue SBuf/RBuf: value members — iresult owns the buffers until wait()/test().
/// Move-only: moving transfers the unique_ptr without relocating the buffers.
template <typename SBuf, typename RBuf>
class [[nodiscard]] iresult<SBuf, RBuf> {
    std::unique_ptr<result<SBuf, RBuf>> result_;
    // Declared after result_ so that it is destroyed first (see the lifetime invariant above).
    mpi::experimental::request request_;

public:
    explicit iresult(SBuf&& sbuf, RBuf&& rbuf)
        : result_(
              std::make_unique<result<SBuf, RBuf>>(
                  result<SBuf, RBuf>{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)}
              )
          ) {}

    iresult(iresult&&) noexcept = default;

    /// Releases this iresult's buffers and takes over `other`'s. Aborts if this iresult's operation is still
    /// outstanding.
    iresult& operator=(iresult&& other) noexcept {
        iresult tmp(std::move(other));
        std::swap(result_, tmp.result_);
        std::swap(request_, tmp.request_);
        return *this; // tmp's destructor checks the old request, then releases the old buffers
    }

    /// For the MPI call site to fill in the request (satisfies convertible_to_mpi_handle_ptr<MPI_Request>).
    MPI_Request* mpi_handle_ptr() noexcept {
        return request_.mpi_handle_ptr();
    }

    // ── Buffer access ─────────────────────────────────────────────────────────

    SBuf& send() {
        return result_->send;
    }
    RBuf& recv() {
        return result_->recv;
    }

    // ── Completion ────────────────────────────────────────────────────────────

    /// Blocks until complete, then returns the result.
    /// Owned members are moved out; borrowed members copy their reference binding (no data copy).
    template <mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    result<SBuf, RBuf> wait(Status&& status = MPI_STATUS_IGNORE) {
        request_.wait(std::forward<Status>(status));
        return std::move(*result_);
    }

    /// Non-blocking completion check. Always returns optional<result<SBuf, RBuf>>.
    ///   Some: operation complete — owned members moved out, ref bindings copied (no data copy).
    ///   nullopt: pending — iresult remains valid for retry or wait().
    template <mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    std::optional<result<SBuf, RBuf>> test(Status&& status = MPI_STATUS_IGNORE) {
        if (request_.test(std::forward<Status>(status)))
            return std::move(*result_);
        return std::nullopt;
    }
};

} // namespace kamping::v2
