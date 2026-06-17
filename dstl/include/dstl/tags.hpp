// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <type_traits>

/// @file
/// Zero-overhead tag types for the grid alltoallv: execution policy (D3) and recv
/// ordering (D5). See DSTL-Alltoallv-Design.md for the rationale behind binding
/// these as compile-time tags rather than runtime enums.

namespace dstl {

// Execution policy (D3)// The three thread models differ in *resources*, not just code, so they are bound
// to the grid_comm type and dispatched via `if constexpr` in the re-bin kernel.

/// MPI_THREAD_SINGLE — serial rearrange, one communicator per dimension.
struct sequential {};

/// MPI_THREAD_FUNNELED — OpenMP-parallel rearrange, one communicator per dimension.
/// Falls back to a serial rebin when the translation unit is not compiled with OpenMP.
struct funneled {};

/// MPI_THREAD_MULTIPLE — like funneled for the rearrange, but additionally requires
/// (and asserts) MPI_THREAD_MULTIPLE so concurrent per-phase MPI can be layered on later.
struct concurrent {};

/// Concept: one of the execution policy tags above.
template <typename T>
concept execution_policy =
    std::is_same_v<T, sequential> || std::is_same_v<T, funneled> || std::is_same_v<T, concurrent>;

// Recv ordering (D5)
/// Default: the recv buffer holds the correct *multiset*, grouped by routing path.
/// This is where the speedup lives — restoring global-source order is extra work.
struct unordered {};

/// Opt-in: recv layout identical to a flat MPI_Alltoallv (per-source recv_counts,
/// elements grouped by increasing global source rank). Carries source labels through
/// every hop and performs a final local stable sort.
struct ordered_by_source {};

/// Concept: one of the ordering tags above.
template <typename T>
concept recv_ordering = std::is_same_v<T, unordered> || std::is_same_v<T, ordered_by_source>;

} // namespace dstl
