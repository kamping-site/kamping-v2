// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <optional>
#include <ranges>

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/arrays.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

/// Blocks until any one operation in `requests` completes (`MPI_Waitany`).
///
/// The completed request is set to `MPI_REQUEST_NULL` (or marked inactive, for persistent requests).
///
/// @param requests A writable contiguous range of `MPI_Request` (see `request_range`).
/// @param status   Optional status out-parameter for the completed operation; omit to use `MPI_STATUS_IGNORE`.
/// @return The index of the completed request, or `std::nullopt` if `requests` contains no active request
///         (`MPI_UNDEFINED`).
template <request_range Requests, convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
std::optional<std::size_t> waitany(Requests&& requests, Status&& status = MPI_STATUS_IGNORE) {
    auto&& reqs  = handle_range(requests);
    int    index = MPI_UNDEFINED;
    int    err =
        MPI_Waitany(static_cast<int>(std::ranges::size(reqs)), std::ranges::data(reqs), &index, handle_ptr(status));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    if (index == MPI_UNDEFINED) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(index);
}

/// Result of `testany`: an optional-like index with an additional *pending* state.
///
/// Exactly one of three states holds:
///   - `pending()`:                  no operation has completed yet; `waitany` would block.
///   - `has_value()`, index `*r`:    `requests[*r]` completed and was set to `MPI_REQUEST_NULL`.
///   - neither:                      `requests` contains no active request (`MPI_UNDEFINED`).
///
/// There is deliberately no `operator bool`: `if (r)` could plausibly mean either "not pending" or "has an
/// index", and guessing wrong in a polling loop spins forever. Call `pending()` or `has_value()` explicitly.
class testany_result {
public:
    /// Constructs from the `flag` and `index` outputs of `MPI_Testany`.
    testany_result(bool flag, int index) noexcept : _flag(flag), _index(index) {}

    /// @return `true` if no operation has completed yet.
    [[nodiscard]] bool pending() const noexcept {
        return !_flag;
    }

    /// @return `true` if an operation completed; its index is then available via `*r` or `value()`.
    [[nodiscard]] bool has_value() const noexcept {
        return _flag && _index != MPI_UNDEFINED;
    }

    /// @return The index of the completed request. Precondition: `has_value()`.
    [[nodiscard]] std::size_t operator*() const noexcept {
        KAMPING_V2_ASSERT(has_value(), "testany_result has no index");
        return static_cast<std::size_t>(_index);
    }

    /// @return The index of the completed request.
    /// @throws std::bad_optional_access if `!has_value()`.
    [[nodiscard]] std::size_t value() const {
        if (!has_value()) {
            throw std::bad_optional_access{};
        }
        return static_cast<std::size_t>(_index);
    }

    /// @return What `waitany` would have returned: the completed index, or `std::nullopt` if there was no active
    ///         request. Precondition: `!pending()`.
    [[nodiscard]] std::optional<std::size_t> index() const noexcept {
        KAMPING_V2_ASSERT(!pending(), "testany_result is pending; there is no waitany result yet");
        if (_index == MPI_UNDEFINED) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(_index);
    }

private:
    bool _flag;  ///< MPI_Testany's flag: false while no operation has completed.
    int  _index; ///< MPI_Testany's index: the completed request, or MPI_UNDEFINED.
};

/// Checks whether any one operation in `requests` has completed (`MPI_Testany`).
///
/// @param requests A writable contiguous range of `MPI_Request` (see `request_range`).
/// @param status   Optional status out-parameter for the completed operation; omit to use `MPI_STATUS_IGNORE`.
/// @return A `testany_result`: pending, the completed index, or no active request.
template <request_range Requests, convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
testany_result testany(Requests&& requests, Status&& status = MPI_STATUS_IGNORE) {
    auto&& reqs  = handle_range(requests);
    int    index = MPI_UNDEFINED;
    int    flag  = 0;
    int    err   = MPI_Testany(
        static_cast<int>(std::ranges::size(reqs)),
        std::ranges::data(reqs),
        &index,
        &flag,
        handle_ptr(status)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return testany_result{flag != 0, index};
}

} // namespace mpi::experimental
