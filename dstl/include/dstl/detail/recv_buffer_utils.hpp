// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#include "kamping/v2/kassert.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "mpi/buffer.hpp"

/// @file
/// Recv-buffer sizing utilities shared by the dstl collectives that deposit a routed/returned result into a
/// caller-provided recv buffer — the grid all-to-all-v (dstl/grid_alltoallv.hpp) and the reversible
/// request_reply (dstl/request_reply.hpp). Both must turn a set of per-source recv counts into a sized
/// buffer plus base write offsets, honouring the kamping v2 deferred-buffer protocol (set_recv_count /
/// set_comm_size / commit_counts / materialize). The logic is identical, so it lives here once.

namespace dstl::detail {

/// Ensure a non-variadic recv buffer has room for `total_recv` elements ahead of a deposit. Resizing is
/// *opt-in*, matching the kamping v2 convention (only deferred buffers are sized; `infer()` likewise only
/// ever calls `set_recv_count` on a `deferred_recv_buf`):
///   * a deferred recv buffer (`views::resize`) records the size via `set_recv_count`; the actual
///     allocation is deferred to the buffer's first `ptr()` access (the alltoallv that receives into it),
///     so there is no eager `materialize` here.
///   * any other buffer — a plain `std::vector`, a span — is assumed pre-sized by the caller and only
///     checked. (Pass `views::resize` to opt into automatic resizing.)
/// A *variadic* deferred buffer is rejected: its per-source counts are unknown here, so a variadic output
/// path sizes such buffers itself (`size_from_source_counts`). `deferred_recv_buf` is satisfied on
/// reference types, so no `remove_cvref` is needed.
template <typename RBuf>
void ensure_recv_capacity(RBuf& rbuf, int total_recv) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(total_recv));
    } else {
        static_assert(
            !kamping::v2::deferred_recv_buf_v<RBuf>,
            "this recv path cannot fill a variadic recv buffer's per-source counts; pass a non-variadic "
            "recv buffer (e.g. views::resize) or use the ordered (variadic) output."
        );
        KAMPING_V2_ASSERT(
            static_cast<std::size_t>(mpi::experimental::count(rbuf)) >= static_cast<std::size_t>(total_recv),
            "non-resizable recv buffer is smaller than the received element total; pass views::resize to opt "
            "into automatic resizing."
        );
    }
}

/// Size `rbuf` from the per-source recv counts and return each source's base write offset (its
/// displacement). For a variadic deferred buffer (`deferred_recv_buf_v`, e.g. `views::auto_recv_v`) the
/// histogram *is* the buffer's `recv_counts`, so we write it straight in, then `commit_counts()` +
/// `materialize()` size the buffer and derive the displacements we hand back — counts and offsets in one
/// pass. For any other buffer the buffer is sized to the total via `ensure_recv_capacity` and the offsets
/// are an exclusive scan of the counts.
template <typename RBuf>
std::vector<int> size_from_source_counts(RBuf& rbuf, std::span<int const> per_source_counts, int total_recv) {
    if constexpr (kamping::v2::deferred_recv_buf_v<RBuf>) {
        rbuf.set_comm_size(static_cast<int>(per_source_counts.size()));
        auto recv_counts = mpi::experimental::counts(rbuf); // mutable per-source counts buffer
        std::ranges::copy(per_source_counts, std::ranges::begin(recv_counts));
        rbuf.commit_counts();
        // Deliberate eager realization — NOT a strict correctness need (the lazy accessors self-realize:
        // auto_displs computes displs on read, resize_v resizes on ptr). But a parallel host-side scatter
        // that follows consumes the buffer's OWN derived displs() + ptr() storage directly, so we force the
        // counts->displs derivation and the resize here, once and single-threaded, before any parallel
        // region — making it race-free regardless of where the accessors are first touched.
        kamping::v2::materialize(rbuf);
        auto recv_displs = mpi::experimental::displs(rbuf); // derived from the counts written above
        return std::vector<int>(std::ranges::begin(recv_displs), std::ranges::end(recv_displs));
    } else {
        ensure_recv_capacity(rbuf, total_recv);
        std::vector<int> displs(per_source_counts.size());
        std::exclusive_scan(per_source_counts.begin(), per_source_counts.end(), displs.begin(), 0);
        return displs;
    }
}

} // namespace dstl::detail
