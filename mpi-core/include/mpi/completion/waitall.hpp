// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <ranges>

#include <mpi.h>

#include "mpi/arrays.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"
#include "mpi/sentinels.hpp"

namespace mpi::experimental {

/// Blocks until all operations in `requests` complete (`MPI_Waitall`).
///
/// On return, every request is set to `MPI_REQUEST_NULL` (or marked inactive, for persistent requests).
/// `MPI_REQUEST_NULL` entries are allowed and ignored.
///
/// @param requests A writable contiguous range of `MPI_Request` (see `request_range`).
/// @param statuses Optional: a writable contiguous range of `MPI_Status` with at least as many entries as
///                 `requests`; `statuses[i]` receives the status of `requests[i]`. Defaults to `statuses_ignore`.
template <request_range Requests, output_array_or_ignore_of<MPI_Status> Statuses = statuses_ignore_t>
void waitall(Requests&& requests, Statuses&& statuses = statuses_ignore_t{}) {
    auto&&      reqs = handle_range(requests);
    std::size_t n    = std::ranges::size(reqs);
    int         err =
        MPI_Waitall(static_cast<int>(n), std::ranges::data(reqs), detail::output_array_ptr<MPI_Status>(statuses, n));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// Checks whether all operations in `requests` have completed (`MPI_Testall`).
///
/// Either all requests complete and are set to `MPI_REQUEST_NULL`, or none of them is modified.
///
/// @param requests A writable contiguous range of `MPI_Request` (see `request_range`).
/// @param statuses Optional: a writable contiguous range of `MPI_Status` with at least as many entries as
///                 `requests`; only written if this returns `true`. Defaults to `statuses_ignore`.
/// @return `true` if all operations have completed.
template <request_range Requests, output_array_or_ignore_of<MPI_Status> Statuses = statuses_ignore_t>
bool testall(Requests&& requests, Statuses&& statuses = statuses_ignore_t{}) {
    auto&&      reqs = handle_range(requests);
    std::size_t n    = std::ranges::size(reqs);
    int         flag = 0;
    int         err  = MPI_Testall(
        static_cast<int>(n),
        std::ranges::data(reqs),
        &flag,
        detail::output_array_ptr<MPI_Status>(statuses, n)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return flag != 0;
}

} // namespace mpi::experimental
