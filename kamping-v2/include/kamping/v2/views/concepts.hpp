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
concept deferred_recv_buf_v = mpi::experimental::has_mpi_counts_mutable<T>
                           && has_commit_counts<T>
                           && has_set_comm_size<T>;

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

} // namespace kamping::v2
