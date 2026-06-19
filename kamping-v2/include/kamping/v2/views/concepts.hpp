// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>

#include <mpi.h>
#include <mpi/buffer.hpp>

namespace kamping::v2 {

/// A buffer whose mpi_displs() are guaranteed to be monotonically non-decreasing
/// (e.g. computed via exclusive_scan). When true, resize_v_view can use the O(1)
/// tight bound displs.back()+counts.back() instead of the O(p) max formula.
template <typename T>
concept has_monotonic_displs = requires(T const& t) {
    { t.displs_monotonic() } -> std::convertible_to<bool>;
};

/// A type that participates in the commit_counts() protocol.
/// Calling commit_counts() signals that the counts buffer has been fully written
/// by MPI and any cached derived state (e.g. displacements) should be invalidated.
template <typename T>
concept has_commit_counts = requires(T& t) { t.commit_counts(); };

/// Recv buffer whose size is not known upfront and is determined by infer().
/// The collective calls set_recv_count(n) after inferring n; the view then
/// lazily resizes on first mpi_data() access.
template <typename T>
concept deferred_recv_buf = requires(T& t, std::ptrdiff_t n) { t.set_recv_count(n); };

template <typename T>
concept has_set_comm_size = requires(T& t, int n) { t.set_comm_size(n); };

/// Recv buffer for variadic collectives whose per-rank counts are not known upfront.
/// The infer() protocol:
///   1. set_comm_size(comm_size)       — pre-allocate the counts buffer
///   2. MPI writes into mpi_counts()   — fill per-rank counts directly
///   3. commit_counts()               — signal counts are ready; invalidate cached state
template <typename T>
concept deferred_recv_buf_v =
    mpi::experimental::has_mpi_counts_mutable<T> && has_commit_counts<T> && has_set_comm_size<T>;

/// A buffer that can be eagerly realized via a `materialize()` member. The
/// deferred receive views (`resize_view`, `resize_v_view`) implement this to
/// perform up front the resize / displacement computation that they otherwise
/// postpone until their data pointer is first queried.
template <typename T>
concept has_materialize = requires(T& t) { t.materialize(); };

/// @brief Forces a deferred receive buffer to realize its backing storage *now*.
///
/// Deferred receive buffers (those built with `views::resize` / `views::resize_v` /
/// `views::auto_recv` / `views::auto_recv_v`) postpone resizing the underlying
/// container and computing displacements until their data pointer is first accessed
/// during the MPI call. Custom collectives sometimes need the buffer fully realized
/// *before* the actual exchange — for instance to read back the final displacements
/// in order to lay out a multi-phase transfer. `materialize` makes that an explicit,
/// public operation instead of relying on the side effect of a `ptr()` access.
///
/// Contract: after `materialize(buf)` returns, `count()`, `ptr()`, `counts()` and
/// `displs()` on `buf` yield final, stable values and the backing storage is sized.
/// The operation is idempotent — calling it repeatedly, or letting the MPI wrapper
/// trigger the lazy realization afterwards, is harmless.
///
/// The `materialize()` member is only invoked on buffers that are *both* materializable
/// and an actual deferred receive buffer; on anything else (a plain, already-concrete
/// buffer, or an unrelated type that merely happens to expose a `materialize()` member)
/// this is a no-op, so it is safe to call unconditionally in generic code.
///
/// Propagates automatically through kamping view layers: an outer wrapper (with_type,
/// GPU/serialization adaptors, …) around a deferred buffer forwards materialize() — see
/// view_interface.hpp — so a nested deferred buffer is realized regardless of how many
/// layers wrap it.
///
/// @param buf The buffer to realize. Mutated in place.
template <typename T>
constexpr void materialize(T&& buf) {
    using B = std::remove_reference_t<T>;
    if constexpr (has_materialize<B> && (deferred_recv_buf<B> || deferred_recv_buf_v<B>)) {
        buf.materialize();
    }
}

/// Send buffer for variadic collectives whose per-rank counts depend on the
/// communicator size (e.g. a sparse or flattened source where ranks not present
/// in the input must receive a count of 0). The infer() protocol calls
/// set_comm_size(comm_size) before reading mpi_counts()/mpi_displs(), giving the
/// buffer a chance to lay out its counts/displs/data across all comm_size ranks.
///
/// Unlike deferred_recv_buf_v, the counts are produced *by the buffer* rather than
/// written by MPI, so no mutable mpi_counts()/commit_counts() is required.
template <typename T>
concept deferred_send_buf_v = mpi::experimental::send_buffer_v<T> && has_set_comm_size<T>;

// ──────────────────────────────────────────────────────────────────────────────
// enable_borrowed_buffer — opt-in trait for non-owning buffer types.
//
// A "borrowed" buffer does not own its data: the underlying memory is managed
// externally and will outlive the buffer object. This mirrors std::ranges::
// enable_borrowed_range. The default covers any std::ranges::borrowed_range
// (e.g. std::span, std::string_view). Kamping views specialize this in their
// own headers to propagate borrowedness from their Base type.
//
// To opt in a custom non-owning buffer type:
//   template <>
//   inline constexpr bool kamping::v2::enable_borrowed_buffer<MyView> = true;
// ──────────────────────────────────────────────────────────────────────────────
template <typename T>
inline constexpr bool enable_borrowed_buffer = std::ranges::borrowed_range<T>;

template <typename T>
concept borrowed_buffer = enable_borrowed_buffer<std::remove_cvref_t<T>>;

// ── supports_matched_probe ─────────────────────────────────────────────────────────
/// Trait controlling whether infer() uses matched probing (MPI_Mprobe / MPI_Mrecv)
/// for a deferred recv buffer.  Defaults to `true`.  Set to `false` for buffer
/// types whose MPI implementation does not support GPU-aware matched receives
/// (e.g. `thrust::device_vector` with Intel MPI); infer() then falls back to a
/// plain MPI_Probe followed by a regular MPI_Recv.
///
/// Propagates automatically through kamping view layers — see view_interface.hpp.
///
/// To opt out a custom buffer type:
/// @code
///   template <>
///   inline constexpr bool kamping::v2::supports_matched_probe<MyGpuBuffer> = false;
/// @endcode
template <typename T>
inline constexpr bool supports_matched_probe = true;

} // namespace kamping::v2
