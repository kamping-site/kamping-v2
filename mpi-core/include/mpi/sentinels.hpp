#pragma once

#include <cstddef>
#include <span>

#include <mpi.h>

/// @file
/// Sentinel buffer types for special MPI data pointer values.
/// All three satisfy the buffer concept member protocol and can be passed
/// directly to any mpi::experimental:: operation.

namespace mpi::experimental {

/// Sentinel buffer that passes `MPI_BOTTOM` as the data pointer.
/// Used with derived datatypes that embed absolute addresses.
/// Not a complete `data_buffer` on its own — compose with views::with_type
/// and views::with_size before passing to an MPI operation.
///
///   v2::send(v2::bottom | views::with_type(my_abs_type) | views::with_size(1), dest, comm);
struct bottom_t {
    static void* mpi_ptr() noexcept { return MPI_BOTTOM; }
};
inline constexpr bottom_t bottom{};

/// Sentinel buffer that passes `MPI_IN_PLACE` as the data pointer.
/// Used for in-place collectives where each rank's data is already at the
/// correct position in the receive buffer. Satisfies `send_buffer` and
/// `recv_buffer`. The receive buffer must be pre-sized; deferred resize is
/// not supported for in-place operations.
struct inplace_t {
    static void*          mpi_ptr()  noexcept { return MPI_IN_PLACE; }
    static std::ptrdiff_t mpi_count() noexcept { return 0; }
    static MPI_Datatype   mpi_type()  noexcept { return MPI_DATATYPE_NULL; }
};
inline constexpr inplace_t inplace{};

/// Sentinel buffer that passes `nullptr` as the data pointer.
/// Used for optional root-only buffers in gather/scatter: non-root ranks
/// pass `null_buf` for the receive/send buffer that MPI ignores on their
/// side. Satisfies `send_buffer` and `recv_buffer`.
///
///   v2::gather(local_data, null_buf, root, comm);  // non-root
struct null_buf_t {
    static void*          mpi_ptr()  noexcept { return nullptr; }
    static std::ptrdiff_t mpi_count() noexcept { return 0; }
    static MPI_Datatype   mpi_type()  noexcept { return MPI_DATATYPE_NULL; }
};
inline constexpr null_buf_t null_buf{};

/// Variadic sentinel recv buffer that passes `nullptr` as the data pointer
/// and empty spans for counts and displacements. Satisfies `recv_buffer_v`.
/// Used for non-root ranks in variadic collectives (e.g. gatherv, scatterv)
/// when explicit per-rank counts are already known and no deferred inference
/// is needed. For inference-triggered resizing on root, use
/// `kamping::v2::auto_null_recv_v` instead.
///
///   v2::gatherv(local_data, null_buf_v, root, comm);  // non-root, explicit counts on root
struct null_buf_v_t {
    static void*                mpi_ptr()    noexcept { return nullptr; }
    static MPI_Datatype         mpi_type()   noexcept { return MPI_DATATYPE_NULL; }
    static std::span<int const> mpi_counts() noexcept { return {}; }
    static std::span<int const> mpi_displs() noexcept { return {}; }
};
inline constexpr null_buf_v_t null_buf_v{};

} // namespace mpi::experimental
