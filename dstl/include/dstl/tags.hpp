// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <type_traits>

/// @file
/// Zero-overhead tag types for the grid alltoallv: execution policy (D3) and recv
/// ordering (D5). See DSTL-Alltoallv-Design.md for the rationale behind binding
/// these as compile-time tags rather than runtime enums.

namespace dstl {

// Execution policy (D3)
//
// These tags select *what is parallelized* (nothing / compute / compute + communication); the MPI
// threading level each one needs is a consequence, not the thing being chosen. The policies differ
// in *resources*, not just code, so they are bound to the grid_comm type and dispatched via
// `if constexpr` in the re-bin kernel.

namespace execution_policy {

/// Sequential: one thread regroups the elements between phases and issues all MPI calls. Works with any MPI thread
/// level.
struct seq {};

/// Parallel computation (the default): the elements are regrouped between phases with OpenMP, but MPI is called from
/// a single thread. Requires at least `MPI_THREAD_FUNNELED`. Without OpenMP, the regrouping is sequential.
struct par {};

/// Parallel computation and communication: like `par`, but MPI is also called concurrently from several threads,
/// each on its own duplicated communicator. Requires `MPI_THREAD_MULTIPLE` (asserted).
struct par_comm {};

} // namespace execution_policy

/// One of the execution policies `seq`, `par` and `par_comm`.
template <typename T>
concept is_execution_policy = std::is_same_v<T, execution_policy::seq> || std::is_same_v<T, execution_policy::par>
                              || std::is_same_v<T, execution_policy::par_comm>;

// Recv ordering (D5)

namespace layout {

/// The receive buffer holds the received elements grouped by routing path, not by source rank (the default). Restoring
/// the order of a flat `alltoallv` is extra work, so prefer this if the order does not matter.
struct unordered {};

/// The receive buffer holds the received elements as a flat `alltoallv` would: grouped by increasing source rank,
/// with per-source counts. The source of each element is carried along and the elements are sorted locally at the
/// end.
struct ordered_by_source {};

} // namespace layout

/// A receive layout: `layout::unordered` or `layout::ordered_by_source`. Used by `grid_alltoallv` and
/// `request_reply`.
template <typename T>
concept is_output_layout = std::is_same_v<T, layout::unordered> || std::is_same_v<T, layout::ordered_by_source>;

} // namespace dstl
