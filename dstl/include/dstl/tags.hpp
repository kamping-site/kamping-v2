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

/// Sequential policy: a single thread performs the re-bin and issues all MPI, with one communicator
/// per dimension. Safe at any MPI threading level (needs no more than MPI_THREAD_SINGLE).
struct seq {};

/// Parallel-compute policy: the re-bin is OpenMP-parallel, but MPI is still issued from a single
/// thread (one communicator per dimension). Requires MPI initialized with at least
/// MPI_THREAD_FUNNELED. Falls back to a serial re-bin when the translation unit is compiled without
/// OpenMP.
struct par {};

/// Parallel-compute-and-communicate policy: like `par` for the re-bin, but MPI is additionally
/// issued concurrently from several threads (one duplicated communicator per thread, as the flat
/// thread_multiple exchange does). Requires — and asserts — MPI initialized with MPI_THREAD_MULTIPLE.
struct par_comm {};

} // namespace execution_policy

/// Concept: one of the execution policy tags above.
template <typename T>
concept is_execution_policy = std::is_same_v<T, execution_policy::seq> ||
                              std::is_same_v<T, execution_policy::par> ||
                              std::is_same_v<T, execution_policy::par_comm>;

// Recv ordering (D5)

namespace layout {

/// Default: the recv buffer holds the correct *multiset*, grouped by routing path.
/// This is where the speedup lives — restoring global-source order is extra work.
struct unordered {};

/// Opt-in: recv layout identical to a flat MPI_Alltoallv (per-source recv_counts,
/// elements grouped by increasing global source rank). Carries source labels through
/// every hop and performs a final local stable sort.
struct ordered_by_source {};

} // namespace layout

/// Concept: a recv-ordering tag — `layout::unordered` or `layout::ordered_by_source`. Shared by the grid
/// alltoallv (`grid_compatible_output_layout`) and the request_reply output (which reuses the same two
/// tags: `unordered` → a plain `recv_buffer`, `ordered_by_source` → a variadic `recv_buffer_v`).
template <typename T>
concept is_output_layout =
    std::is_same_v<T, layout::unordered> || std::is_same_v<T, layout::ordered_by_source>;

} // namespace dstl
