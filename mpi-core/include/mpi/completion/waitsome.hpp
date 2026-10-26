// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <optional>
#include <ranges>

#include <mpi.h>

#include "mpi/arrays.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"
#include "mpi/sentinels.hpp"

namespace mpi::experimental {

namespace detail {
/// Shared implementation of waitsome/testsome; `Op` is `MPI_Waitsome` or `MPI_Testsome`.
template <typename Op, typename Requests, typename Indices, typename Statuses>
std::optional<std::size_t> some_impl(Op op, Requests& requests, Indices& indices, Statuses& statuses) {
    auto&&      reqs     = handle_range(requests);
    std::size_t n        = std::ranges::size(reqs);
    int         outcount = MPI_UNDEFINED;
    int         err =
        op(static_cast<int>(n),
           std::ranges::data(reqs),
           &outcount,
           output_array_ptr<int>(indices, n),
           output_array_ptr<MPI_Status>(statuses, n));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    if (outcount == MPI_UNDEFINED) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(outcount);
}
} // namespace detail

/// Blocks until at least one operation in `requests` completes (`MPI_Waitsome`).
///
/// Completed requests are set to `MPI_REQUEST_NULL` (or marked inactive, for persistent requests).
///
/// @param requests A writable contiguous range of `MPI_Request` (see `request_range`).
/// @param indices  A writable contiguous range of `int` with at least as many entries as `requests`;
///                 the first `n` entries receive the indices of the completed requests.
/// @param statuses Optional: a writable contiguous range of `MPI_Status` with at least as many entries as
///                 `requests`, written in completion order: `statuses[k]` belongs to `requests[indices[k]]`, not to
///                 `requests[k]`. Defaults to `statuses_ignore`.
/// @return The number `n` of completed requests, or `std::nullopt` if `requests` contains no active request
///         (`MPI_UNDEFINED`).
template <
    request_range                         Requests,
    output_array_of<int>                  Indices,
    output_array_or_ignore_of<MPI_Status> Statuses = statuses_ignore_t>
std::optional<std::size_t> waitsome(Requests&& requests, Indices&& indices, Statuses&& statuses = statuses_ignore_t{}) {
    return detail::some_impl(MPI_Waitsome, requests, indices, statuses);
}

/// Checks which operations in `requests` have completed, without blocking (`MPI_Testsome`).
///
/// Same contract as `waitsome`, except that it returns `0` if no operation has completed yet.
///
/// @return The number of completed requests (possibly `0`), or `std::nullopt` if `requests` contains no active
///         request (`MPI_UNDEFINED`).
template <
    request_range                         Requests,
    output_array_of<int>                  Indices,
    output_array_or_ignore_of<MPI_Status> Statuses = statuses_ignore_t>
std::optional<std::size_t> testsome(Requests&& requests, Indices&& indices, Statuses&& statuses = statuses_ignore_t{}) {
    return detail::some_impl(MPI_Testsome, requests, indices, statuses);
}

} // namespace mpi::experimental
