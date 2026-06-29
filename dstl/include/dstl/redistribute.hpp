// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <ranges>
#include <vector>

#include <mpi.h>

#include "kamping/v2/collectives/allgather.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/p2p/irecv.hpp"
#include "kamping/v2/p2p/isend.hpp"
#include "kamping/v2/sentinels.hpp"
#include "mpi/buffer.hpp"
#include "mpi/handle.hpp"
#include "mpi/mpi_span.hpp"

namespace dstl {

/// @brief Redistributes data between ranks according to globally agreed send/recv windows.
///
/// Each rank owns a contiguous slice of a conceptual global send buffer (sbuf) and wants to
/// receive a contiguous slice of a conceptual global recv buffer (rbuf). The two global
/// arrays are the same data in flight: what rank i sends at global offset [s_i, s_i+k)
/// lands in whoever's recv window covers that range.
///
/// Algorithm:
///   1. Allgather all per-rank send/recv counts to derive global displacement arrays.
///   2. For each rank, binary-search for the set of remote ranks whose send (or recv) window
///      overlaps this rank's recv (or send) window, and issue one irecv/isend per overlap.
///   3. MPI_Waitall.
///
/// No additional data copies beyond the MPI transfers are performed.
template <
    mpi::experimental::send_buffer                           SBuf,
    mpi::experimental::recv_buffer                           RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>   Comm = MPI_Comm,
    mpi::experimental::tag                                   Tag  = int>
void redistribute(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD, Tag tag = 0) {
    kamping::v2::comm_view cv{mpi::experimental::handle(comm)};
    int const size = cv.size();
    int const rank = cv.rank();

    // Interleaved [send0, recv0, ..., send_{p-1}, recv_{p-1}]; sentinel slots stay 0.
    std::vector<int> counts(static_cast<std::size_t>(size + 1) * 2);
    counts[2 * static_cast<std::size_t>(rank)]       = static_cast<int>(mpi::experimental::count(sbuf));
    counts[2 * static_cast<std::size_t>(rank) + 1]   = static_cast<int>(mpi::experimental::count(rbuf));
    kamping::v2::allgather(kamping::v2::inplace, counts | std::views::take(2 * size), cv);

    // Single-pass prefix sum over interleaved slots; sentinel slots receive the totals.
    for (int send_sum = 0, recv_sum = 0, i = 0; i <= size; ++i) {
        auto const k = static_cast<std::size_t>(i);
        int const  sc = counts[2 * k];
        int const  rc = counts[2 * k + 1];
        counts[2 * k]     = send_sum; send_sum += sc;
        counts[2 * k + 1] = recv_sum; recv_sum += rc;
    }

    // Lazy views into the interleaved displacement array — no extra allocation.
    auto send_displs = std::views::iota(0, size + 1) | std::views::transform([&](int i) {
        return counts[static_cast<std::size_t>(2 * i)];
    });
    auto recv_displs = std::views::iota(0, size + 1) | std::views::transform([&](int i) {
        return counts[static_cast<std::size_t>(2 * i + 1)];
    });

    MPI_Aint send_lb{}, send_extent{}, recv_lb{}, recv_extent{};
    MPI_Type_get_extent(mpi::experimental::type(sbuf), &send_lb, &send_extent);
    MPI_Type_get_extent(mpi::experimental::type(rbuf), &recv_lb, &recv_extent);

    auto const* send_base = static_cast<std::byte const*>(static_cast<void const*>(mpi::experimental::ptr(sbuf)));
    auto*       recv_base = static_cast<std::byte*>(static_cast<void*>(mpi::experimental::ptr(rbuf)));

    std::vector<MPI_Request> requests;
    requests.reserve(2 * static_cast<std::size_t>(size));

    int const recv_begin = recv_displs[rank];
    int const recv_end   = recv_displs[rank + 1];
    if (recv_begin < recv_end) {
        auto const first = std::ranges::upper_bound(send_displs, recv_begin) - send_displs.begin() - 1;
        auto const last  = std::ranges::lower_bound(send_displs, recv_end)   - send_displs.begin() - 1;
        MPI_Aint recv_offset = 0;
        for (auto j : std::views::iota(first, last + 1)) {
            int const n = std::min(recv_end, send_displs[j + 1]) - std::max(recv_begin, send_displs[j]);
            kamping::v2::irecv(
                requests.emplace_back(),
                mpi::experimental::mpi_span(recv_base + recv_offset * recv_extent, n, mpi::experimental::type(rbuf)),
                static_cast<int>(j),
                tag,
                cv
            );
            recv_offset += n;
        }
    }

    int const send_begin = send_displs[rank];
    int const send_end   = send_displs[rank + 1];
    if (send_begin < send_end) {
        auto const first = std::ranges::upper_bound(recv_displs, send_begin) - recv_displs.begin() - 1;
        auto const last  = std::ranges::lower_bound(recv_displs, send_end)   - recv_displs.begin() - 1;
        MPI_Aint send_offset = 0;
        for (auto j : std::views::iota(first, last + 1)) {
            int const n = std::min(send_end, recv_displs[j + 1]) - std::max(send_begin, recv_displs[j]);
            kamping::v2::isend(
                requests.emplace_back(),
                mpi::experimental::mpi_cspan(send_base + send_offset * send_extent, n, mpi::experimental::type(sbuf)),
                static_cast<int>(j),
                tag,
                cv
            );
            send_offset += n;
        }
    }

    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
}

} // namespace dstl
