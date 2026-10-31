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

/// Sentinel buffer that passes `MPI_BOTTOM` as the data pointer.
/// Used with derived datatypes that embed absolute addresses.
/// Not a complete `data_buffer` on its own — compose with views::with_type
/// and views::with_count before passing to an MPI operation.
///
///   v2::send(v2::bottom | views::with_type(my_abs_type) | views::with_count(1), dest, comm);
struct bottom_t {
    static void* mpi_ptr() noexcept {
        return MPI_BOTTOM;
    }
};
/// The @ref bottom_t sentinel.
inline constexpr bottom_t bottom{};

/// Sentinel buffer that passes `MPI_IN_PLACE` as the data pointer.
/// Used for in-place collectives where each rank's data is already at the
/// correct position in the receive buffer. Satisfies `send_buffer` and
/// `recv_buffer`. The receive buffer must be pre-sized; deferred resize is
/// not supported for in-place operations.
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
/// The @ref inplace_t sentinel.
inline constexpr inplace_t inplace{};

/// Sentinel buffer that passes `nullptr` as the data pointer.
/// Used for optional root-only buffers in gather/scatter: non-root ranks
/// pass `null_buf` for the receive/send buffer that MPI ignores on their
/// side. Also usable as an empty message in point-to-point calls, which is why
/// the datatype is `MPI_BYTE`: MPI rejects `MPI_DATATYPE_NULL` there even for
/// count 0. Satisfies `send_buffer` and `recv_buffer`.
///
///   v2::gather(local_data, null_buf, root, comm);  // non-root
///   v2::send(null_buf, dest, tag, comm);           // zero-byte signal
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
/// The @ref null_buf_t sentinel.
inline constexpr null_buf_t null_buf{};

/// Variadic sentinel buffer that passes `nullptr` as the data pointer
/// and empty spans for counts and displacements. Satisfies `send_buffer_v` and `recv_buffer_v`.
/// Used for non-root ranks in variadic collectives (e.g. gatherv, scatterv)
/// when explicit per-rank counts are already known and no deferred inference
/// is needed. For inference-triggered resizing on root, use
/// `kamping::v2::auto_null_recv_v` instead.
///
///   v2::gatherv(local_data, null_buf_v, root, comm);  // non-root, explicit counts on root
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
/// The @ref null_buf_v_t sentinel.
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
