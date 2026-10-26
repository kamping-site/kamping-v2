// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/detail/handle_iterator.hpp"
#include "mpi/handle.hpp"
#include "mpi/status.hpp"

namespace mpi::experimental {

/// @brief Growable array of MPI statuses that mirrors `std::vector`, but whose element access yields
/// `status_view`s (or `const_status_view`s, through a const vector) instead of raw `MPI_Status`es.
///
/// The statuses are stored contiguously, so the whole array can be passed as the statuses argument of the
/// multiple-completion functions (`waitall`, `waitsome`, …) without copying; it exposes them via
/// `mpi_handle_range()` rather than `data()`, since its elements as a range are views, not `MPI_Status`es.
///
///   request_vector requests;
///   status_vector  statuses(requests.size());
///   waitall(requests, statuses);
///   int source = statuses[0].source();
///
/// Unlike request_vector, a status is a plain value with no MPI resource attached, so status_vector is copyable
/// and dropping elements needs no completion. Views are invalidated by reallocation, as with `std::vector`.
class status_vector {
public:
    using value_type      = status_view;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using iterator        = detail::handle_iterator<MPI_Status, status_view>;
    using const_iterator  = detail::handle_iterator<MPI_Status const, const_status_view>;

    status_vector() = default;

    /// @brief Constructs `n` statuses. Their fields are undefined until filled in by MPI.
    explicit status_vector(size_type n) : _statuses(n) {}

    // ── Capacity ────────────────────────────────────────────────────────────

    [[nodiscard]] size_type size() const noexcept {
        return _statuses.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return _statuses.empty();
    }

    [[nodiscard]] size_type capacity() const noexcept {
        return _statuses.capacity();
    }

    /// @brief Reserves storage for `n` statuses. Invalidates all views if it reallocates.
    void reserve(size_type n) {
        _statuses.reserve(n);
    }

    /// @brief Resizes to `n` statuses. Invalidates all views if it reallocates.
    void resize(size_type n) {
        _statuses.resize(n);
    }

    void clear() noexcept {
        _statuses.clear();
    }

    // ── Element access ──────────────────────────────────────────────────────

    /// @return A view of the status at index `i`. Precondition: `i < size()`.
    status_view operator[](size_type i) noexcept {
        KAMPING_V2_ASSERT(i < size(), "status_vector index out of range");
        return status_view{&_statuses[i]};
    }

    /// @return A read-only view of the status at index `i`. Precondition: `i < size()`.
    const_status_view operator[](size_type i) const noexcept {
        KAMPING_V2_ASSERT(i < size(), "status_vector index out of range");
        return const_status_view{&_statuses[i]};
    }

    status_view front() noexcept {
        return (*this)[0];
    }

    const_status_view front() const noexcept {
        return (*this)[0];
    }

    status_view back() noexcept {
        return (*this)[size() - 1];
    }

    const_status_view back() const noexcept {
        return (*this)[size() - 1];
    }

    iterator begin() noexcept {
        return iterator{_statuses.data()};
    }

    iterator end() noexcept {
        return iterator{_statuses.data() + _statuses.size()};
    }

    const_iterator begin() const noexcept {
        return const_iterator{_statuses.data()};
    }

    const_iterator end() const noexcept {
        return const_iterator{_statuses.data() + _statuses.size()};
    }

    // ── MPI interop ─────────────────────────────────────────────────────────

    /// @return The underlying contiguous `MPI_Status` array, for the multiple-completion functions
    ///         (see `convertible_to_mpi_handle_range`).
    [[nodiscard]] std::span<MPI_Status> mpi_handle_range() noexcept {
        return _statuses;
    }

private:
    std::vector<MPI_Status> _statuses;
};

static_assert(std::random_access_iterator<status_vector::iterator>);
static_assert(std::random_access_iterator<status_vector::const_iterator>);
static_assert(std::ranges::random_access_range<status_vector>);
static_assert(std::ranges::random_access_range<status_vector const>);
static_assert(!std::ranges::contiguous_range<status_vector>);
static_assert(convertible_to_mpi_handle_range<status_vector, MPI_Status>);
static_assert(!convertible_to_mpi_handle_range<status_vector const, MPI_Status>);

} // namespace mpi::experimental
