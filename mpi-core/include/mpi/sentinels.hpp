// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <span>

#include <mpi.h>

/// @file
/// Sentinel types for special MPI pointer values.
/// The buffer sentinels (bottom, inplace, null_buf, null_buf_v) satisfy the buffer concept member protocol and can
/// be passed directly to any mpi::experimental:: operation. The output array sentinels (statuses_ignore) stand in
/// for an ignorable output array argument (see `output_array_or_ignore_of` in mpi/arrays.hpp).

namespace mpi::experimental {

/// Type of @ref bottom.
struct bottom_t {
    static void* mpi_ptr() noexcept {
        return MPI_BOTTOM;
    }
};
/// A buffer at `MPI_BOTTOM`, for derived datatypes that contain absolute addresses.
///
/// Not a complete buffer on its own: add the datatype and the count with views.
///
/// @code
/// v2::send(v2::bottom | views::with_type(absolute_type) | views::with_count(1), dest, comm);
/// @endcode
inline constexpr bottom_t bottom{};

/// Type of @ref inplace.
struct inplace_t {
    static void* mpi_ptr() noexcept {
        return MPI_IN_PLACE;
    }
    static std::ptrdiff_t mpi_count() noexcept {
        return 0;
    }
    static MPI_Datatype mpi_type() noexcept {
        return MPI_DATATYPE_NULL;
    }
};
/// `MPI_IN_PLACE`: passed as the send buffer of a collective, the receive buffer holds the input and receives the
/// result.
///
/// The receive buffer must already have the right size; it cannot be resized for an in-place operation.
///
/// @code
/// int sum = local;
/// v2::allreduce(v2::inplace, views::ref_single(sum), comm);
/// @endcode
inline constexpr inplace_t inplace{};

/// Type of @ref null_buf.
//
// The datatype is `MPI_BYTE` because MPI rejects `MPI_DATATYPE_NULL` in point-to-point calls even for count 0.
struct null_buf_t {
    static void* mpi_ptr() noexcept {
        return nullptr;
    }
    static std::ptrdiff_t mpi_count() noexcept {
        return 0;
    }
    static MPI_Datatype mpi_type() noexcept {
        return MPI_BYTE;
    }
};
/// An empty buffer: for the buffer of a rooted collective that only the root uses, or as an empty message.
///
/// The non-root shorthands of `gather`, `scatter` and `reduce` pass it for you.
///
/// @code
/// v2::gather(local_data, v2::null_buf, root, comm);  // on a non-root rank
/// v2::send(v2::null_buf, dest, tag, comm);           // a message without data
/// @endcode
inline constexpr null_buf_t null_buf{};

/// Type of @ref null_buf_v.
struct null_buf_v_t {
    static void* mpi_ptr() noexcept {
        return nullptr;
    }
    static MPI_Datatype mpi_type() noexcept {
        return MPI_DATATYPE_NULL;
    }
    static std::span<int const> mpi_counts() noexcept {
        return {};
    }
    static std::span<int const> mpi_displs() noexcept {
        return {};
    }
};
/// An empty variable-size buffer, for the buffer of `gatherv` or `scatterv` that only the root uses, if the root
/// provides its counts. If the root's receive buffer obtains its counts from the others (e.g. `auto_recv_v`), non-root
/// ranks pass `auto_null_recv_v()` instead.
///
/// @code
/// v2::gatherv(local_data, v2::null_buf_v, root, comm);  // on a non-root rank
/// @endcode
inline constexpr null_buf_v_t null_buf_v{};

// ─── Output array sentinels ──────────────────────────────────────────────────

/// Sentinel output array that passes `MPI_STATUSES_IGNORE`. The default for the statuses argument of `waitall`,
/// `testall`, `waitsome` and `testsome`; pass a range of `MPI_Status` instead to receive the statuses.
///
///   waitall(requests);                   // statuses ignored
///   waitall(requests, statuses_ignore);  // the same, spelled out
///
/// Deliberately not convertible to a single `MPI_Status*` (no `mpi_handle_ptr()`), so it cannot be passed as the
/// status of `wait`/`test`, where `MPI_STATUS_IGNORE` is the matching value.
struct statuses_ignore_t {
    static MPI_Status* mpi_ptr() noexcept {
        return MPI_STATUSES_IGNORE;
    }
};
/// The @ref statuses_ignore_t sentinel.
inline constexpr statuses_ignore_t statuses_ignore{};

} // namespace mpi::experimental
