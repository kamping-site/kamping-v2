// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <numeric>
#include <ranges>
#include <vector>

#include "mpi/buffer.hpp"

/// @file
/// dstl::exchange_layout / dstl::reverse_layout — the retained per-rank routing of an all-to-all-v, and its
/// transpose. After a forward `kamping::v2::alltoallv` into an auto-managed variadic recv buffer, both
/// buffers carry the negotiated per-rank counts and displacements; `exchange_layout` snapshots them so the
/// exchange can be replayed in reverse with NO further count negotiation (DSTL-Request-Reply-Design R1/R2).
/// `reverse_layout` transposes a layout (send <-> recv); feeding its counts/displs into a second
/// `kamping::v2::alltoallv` ships a 1:1 reply set back along the inverse route.

namespace dstl {

/// The per-rank send and recv counts and displacements of an all-to-all-v, in element units (one entry per
/// rank). `send_*` describe this rank's send (per destination), `recv_*` its recv (per source).
struct exchange_layout {
    std::vector<int> send_counts;
    std::vector<int> send_displs;
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;

    exchange_layout() = default;

    /// Snapshot the layout from the send and recv buffers of a *completed* variadic all-to-all-v. After a
    /// forward `kamping::v2::alltoallv` into an auto-managed recv buffer (`views::auto_recv_v`), the send
    /// buffer carries the send layout and the recv buffer the negotiated recv layout (counts + derived
    /// displs); both are copied out here so they outlive the buffers.
    template <mpi::experimental::send_buffer_v SBuf, mpi::experimental::recv_buffer_v RBuf>
    exchange_layout(SBuf const& sbuf, RBuf const& rbuf) {
        snapshot(send_counts, mpi::experimental::counts(sbuf));
        snapshot(send_displs, mpi::experimental::displs(sbuf));
        snapshot(recv_counts, mpi::experimental::counts(rbuf));
        snapshot(recv_displs, mpi::experimental::displs(rbuf));
    }

    /// Total elements sent (== `sum(send_counts)`).
    [[nodiscard]] int send_total() const {
        return std::reduce(send_counts.begin(), send_counts.end(), 0);
    }
    /// Total elements received (== `sum(recv_counts)`).
    [[nodiscard]] int recv_total() const {
        return std::reduce(recv_counts.begin(), recv_counts.end(), 0);
    }

private:
    template <typename R>
    static void snapshot(std::vector<int>& dst, R&& range) {
        dst.assign(std::ranges::begin(range), std::ranges::end(range));
    }
};

/// The transpose of a layout: the reverse exchange's send layout is the forward recv layout and vice versa.
/// Attaching the result's counts/displs to the buffers of a second `kamping::v2::alltoallv` replays the
/// exchange backward — with both layouts already known, no count negotiation is needed (R1/R2).
[[nodiscard]] inline exchange_layout reverse_layout(exchange_layout const& layout) {
    exchange_layout reversed;
    reversed.send_counts = layout.recv_counts;
    reversed.send_displs = layout.recv_displs;
    reversed.recv_counts = layout.send_counts;
    reversed.recv_displs = layout.send_displs;
    return reversed;
}

} // namespace dstl
