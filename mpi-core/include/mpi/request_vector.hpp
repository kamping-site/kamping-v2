// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/completion/waitall.hpp" // not used here, but waitall(requests) is how a request_vector is completed
#include "mpi/detail/handle_iterator.hpp"
#include "mpi/request.hpp"

namespace mpi::experimental {

/// @brief Owning, growable array of MPI requests that mirrors `std::vector`, but whose element access yields
/// `request_view`s instead of raw `MPI_Request`s.
///
/// The requests are stored contiguously, so the whole array can be passed to the multiple-completion
/// functions (`waitall`, `testany`, `waitsome`, …) without copying; it exposes them via `mpi_handle_range()`
/// rather than `data()`, since its elements as a range are `request_view`s, not `MPI_Request`s.
///
/// Typical use is to hand a fresh slot straight to a non-blocking operation:
///
///   request_vector requests;
///   isend(buf, dest, tag, comm, requests.emplace_back());
///   waitall(requests);
///
/// Ownership: move-only. Like `request`, the vector never completes a request implicitly: every request must be
/// completed explicitly (e.g. `waitall(requests)`) or handed off with `release()`/`extract()` before it is dropped.
/// Dropping an active request prints a diagnostic and calls `std::terminate()`. This applies to the destructor and
/// move assignment, and equally to the modifiers that remove slots: `clear()`, shrinking `resize()` and `pop_back()`.
///
/// Invalidation follows `std::vector`: a `request_view` points into the internal storage, so it is invalidated by
/// any reallocation (`emplace_back`/`push_back` beyond `capacity()`, `reserve`, `resize`) and by `erase_null()`.
/// The `MPI_Request` handles themselves are plain values and survive reallocation unchanged.
///
/// There is no const element access: a `request_view` must be able to complete (and thus write) its request.
class request_vector {
public:
    using value_type      = request_view;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using iterator        = detail::handle_iterator<MPI_Request, request_view>;

    request_vector() = default;

    /// @brief Constructs `n` slots, each `MPI_REQUEST_NULL`.
    explicit request_vector(size_type n) : _requests(n, MPI_REQUEST_NULL) {}

    request_vector(request_vector const&)            = delete;
    request_vector& operator=(request_vector const&) = delete;

    request_vector(request_vector&& other) noexcept : _requests(std::exchange(other._requests, {})) {}

    /// @brief Takes over the requests of `other`. Terminates if this still holds an active request.
    request_vector& operator=(request_vector&& other) noexcept {
        terminate_if_active(_requests);
        _requests = std::exchange(other._requests, {});
        return *this;
    }

    /// @brief Terminates if any request is still active (see class documentation).
    ~request_vector() {
        terminate_if_active(_requests);
    }

    // ── Capacity ────────────────────────────────────────────────────────────

    [[nodiscard]] size_type size() const noexcept {
        return _requests.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return _requests.empty();
    }

    [[nodiscard]] size_type capacity() const noexcept {
        return _requests.capacity();
    }

    /// @brief Reserves storage for `n` requests. Invalidates all views if it reallocates.
    void reserve(size_type n) {
        _requests.reserve(n);
    }

    /// @brief Resizes to `n` slots. New slots are `MPI_REQUEST_NULL`.
    ///
    /// Terminates if a dropped slot (index `>= n`) still holds an active request; complete those first.
    void resize(size_type n) {
        if (n < _requests.size()) {
            terminate_if_active(std::span{_requests}.subspan(n));
        }
        _requests.resize(n, MPI_REQUEST_NULL);
    }

    /// @brief Removes all slots. Terminates if any request is still active; complete them first (e.g. `waitall`).
    void clear() {
        terminate_if_active(_requests);
        _requests.clear();
    }

    // ── Modifiers ───────────────────────────────────────────────────────────

    /// @brief Appends a `MPI_REQUEST_NULL` slot and returns a view of it, ready to be passed to a non-blocking
    /// operation as its request output parameter.
    request_view emplace_back() {
        return request_view{_requests.emplace_back(MPI_REQUEST_NULL)};
    }

    /// @brief Appends the raw handle `req`, taking over responsibility for completing it.
    ///
    /// The caller must not complete `req` itself afterwards: once the vector completes it, MPI frees it and any
    /// remaining copy of the handle is stale.
    void push_back(MPI_Request req) {
        _requests.push_back(req);
    }

    /// @brief Moves the owning request `req` into the vector; `req` is left null.
    void push_back(request&& req) {
        _requests.push_back(MPI_REQUEST_NULL);
        _requests.back() = req.release(); // release only after the slot exists, so a bad_alloc loses nothing
    }

    /// @brief Gives up responsibility for the request at index `i` and returns it as a raw handle.
    ///
    /// The slot becomes `MPI_REQUEST_NULL` but is not removed, so other indices and views stay valid; remove empty
    /// slots later with `erase_null()`. The caller must complete the returned request. Precondition: `i < size()`.
    [[nodiscard]] MPI_Request release(size_type i) noexcept {
        KAMPING_V2_ASSERT(i < size(), "request_vector index out of range");
        return std::exchange(_requests[i], MPI_REQUEST_NULL);
    }

    /// @brief Moves the request at index `i` out into an owning `request`.
    ///
    /// Like `release(i)`, the slot becomes `MPI_REQUEST_NULL` and is not removed. Precondition: `i < size()`.
    [[nodiscard]] request extract(size_type i) noexcept {
        return request{release(i)};
    }

    /// @brief Removes the last slot. Terminates if it still holds an active request. Precondition: `!empty()`.
    void pop_back() {
        KAMPING_V2_ASSERT(!empty(), "pop_back() on an empty request_vector");
        terminate_if_active(std::span{_requests}.last(1));
        _requests.pop_back();
    }

    /// @brief Removes all `MPI_REQUEST_NULL` slots (completed or never started), keeping the order of the rest.
    ///
    /// Useful in polling loops after `testsome`/`waitsome`. Inactive persistent requests are not
    /// `MPI_REQUEST_NULL` and are kept. Invalidates all views.
    /// @return The number of removed slots.
    size_type erase_null() {
        return std::erase(_requests, MPI_REQUEST_NULL);
    }

    // ── Element access ──────────────────────────────────────────────────────

    /// @return A view of the request at index `i`. Precondition: `i < size()`.
    request_view operator[](size_type i) noexcept {
        KAMPING_V2_ASSERT(i < size(), "request_vector index out of range");
        return request_view{_requests[i]};
    }

    request_view front() noexcept {
        return (*this)[0];
    }

    request_view back() noexcept {
        return (*this)[size() - 1];
    }

    iterator begin() noexcept {
        return iterator{_requests.data()};
    }

    iterator end() noexcept {
        return iterator{_requests.data() + _requests.size()};
    }

    // ── MPI interop ─────────────────────────────────────────────────────────

    /// @return The underlying contiguous `MPI_Request` array, for the multiple-completion functions
    ///         (see `convertible_to_mpi_handle_range`).
    [[nodiscard]] std::span<MPI_Request> mpi_handle_range() noexcept {
        return _requests;
    }

private:
    static void terminate_if_active(std::span<MPI_Request const> requests) noexcept {
        if (std::ranges::any_of(requests, [](MPI_Request r) { return r != MPI_REQUEST_NULL; })) {
            detail::terminate_on_active_request("mpi::experimental::request_vector");
        }
    }

    std::vector<MPI_Request> _requests;
};

static_assert(std::random_access_iterator<request_vector::iterator>);
static_assert(std::ranges::random_access_range<request_vector>);
static_assert(std::ranges::sized_range<request_vector>);
static_assert(!std::ranges::contiguous_range<request_vector>);
static_assert(convertible_to_mpi_handle_range<request_vector, MPI_Request>);
static_assert(!convertible_to_mpi_handle_ptr<request_vector, MPI_Request>);

} // namespace mpi::experimental
