// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <iterator>
#include <ranges>
#include <vector>

#include <mpi.h>

#include "kamping/v2/comm.hpp"
#include "kamping/v2/kassert.hpp"
#include "kamping/v2/p2p/sendrecv.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "kamping/v2/views/with_type_view.hpp"
#include "mpi/buffer.hpp"
#include "mpi/handle.hpp"

namespace dstl {

/// @brief Shifts the distributed range left by n positions globally.
///
/// Distributed equivalent of std::shift_left applied to the concatenation of all local
/// ranges across the communicator. Elements at global position i are moved to global
/// position i - n. The last n elements of the last rank are value-initialized.
///
/// Each rank performs a blocking sendrecv with its right neighbor to exchange the n
/// boundary elements, then applies a local std::shift_left. A staging buffer of n
/// elements holds the received data.
///
/// @pre 0 <= n <= std::ranges::ssize(r) on every rank in comm.
///      When n > local range size, alltoallv is needed instead.
///
/// @returns Per-rank subrange of r indicating the locally valid portion after the shift.
///   On every rank except the last this is the full local range.
///   On the last rank it covers only the first (ssize(r) - n) elements.
template <
    std::ranges::contiguous_range                           R,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
    requires mpi::experimental::send_recv_buffer<R>
std::ranges::borrowed_subrange_t<R>
shift_left(R&& r, std::ranges::range_difference_t<R> n, Comm const& comm = MPI_COMM_WORLD) {
    using T          = std::ranges::range_value_t<R>;
    using diff_t     = std::ranges::range_difference_t<R>;
    auto const first = std::ranges::begin(r);
    auto const last  = std::ranges::end(r);
    diff_t const local_size = std::ranges::ssize(r);

    if (n <= 0) {
        return {first, last};
    }
    KAMPING_V2_ASSERT(n <= local_size, "shift_left: n must not exceed local range size on any rank");

    kamping::v2::comm_view cv{mpi::experimental::handle(comm)};
    int const rank  = cv.rank();
    int const p     = cv.size();
    int const left  = (rank > 0)     ? rank - 1 : MPI_PROC_NULL;
    int const right = (rank < p - 1) ? rank + 1 : MPI_PROC_NULL;

    // Capture the MPI type before take() strips it from the view chain.
    MPI_Datatype const mpi_type = mpi::experimental::type(r);

    // For n == 1 avoid a heap allocation by using a stack scalar as the staging buffer.
    // The general path uses a heap-allocated vector for arbitrary n.
    // The recv buffer always carries mpi_type so custom MPI types round-trip correctly.
    if (n == 1) {
        T staging_val{};
        kamping::v2::sendrecv(
            r | std::views::take(1) | kamping::v2::views::with_type(mpi_type),
            left,
            kamping::v2::views::ref_single(staging_val) | kamping::v2::views::with_type(mpi_type),
            right,
            comm
        );
        std::shift_left(first, last, 1);
        *(first + (local_size - 1)) = std::move(staging_val);
    } else {
        std::vector<T> staging(static_cast<std::size_t>(n));
        kamping::v2::sendrecv(
            r | std::views::take(n) | kamping::v2::views::with_type(mpi_type),
            left,
            staging | kamping::v2::views::with_type(mpi_type),
            right,
            comm
        );
        std::shift_left(first, last, n);
        std::ranges::copy(staging, first + (local_size - n));
    }

    if (right == MPI_PROC_NULL) {
        // Last rank: only the first (local_size - n) elements are from the global range.
        return {first, first + std::max(diff_t{0}, local_size - n)};
    }
    return {first, last};
}

/// @brief Shifts the distributed range right by n positions globally.
///
/// Distributed equivalent of std::shift_right applied to the concatenation of all local
/// ranges across the communicator. Elements at global position i are moved to global
/// position i + n. The first n elements of rank 0 are value-initialized.
///
/// Each rank performs a blocking sendrecv with its left neighbor to exchange the n
/// boundary elements, then applies a local std::shift_right. A staging buffer of n
/// elements holds the received data.
///
/// @pre 0 <= n <= std::ranges::ssize(r) on every rank in comm.
///      When n > local range size, alltoallv is needed instead.
///
/// @returns Per-rank subrange of r indicating the locally valid portion after the shift.
///   On every rank except the first this is the full local range.
///   On rank 0 it covers only the last (ssize(r) - n) elements, starting at begin + n.
template <
    std::ranges::contiguous_range                           R,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
    requires mpi::experimental::send_recv_buffer<R>
std::ranges::borrowed_subrange_t<R>
shift_right(R&& r, std::ranges::range_difference_t<R> n, Comm const& comm = MPI_COMM_WORLD) {
    using T          = std::ranges::range_value_t<R>;
    using diff_t     = std::ranges::range_difference_t<R>;
    auto const first = std::ranges::begin(r);
    auto const last  = std::ranges::end(r);
    diff_t const local_size = std::ranges::ssize(r);

    if (n <= 0) {
        return {first, last};
    }
    KAMPING_V2_ASSERT(n <= local_size, "shift_right: n must not exceed local range size on any rank");

    kamping::v2::comm_view cv{mpi::experimental::handle(comm)};
    int const rank  = cv.rank();
    int const p     = cv.size();
    int const left  = (rank > 0)     ? rank - 1 : MPI_PROC_NULL;
    int const right = (rank < p - 1) ? rank + 1 : MPI_PROC_NULL;

    // Capture the MPI type before drop() strips it from the view chain.
    MPI_Datatype const mpi_type = mpi::experimental::type(r);

    // For n == 1 avoid a heap allocation by using a stack scalar as the staging buffer.
    // The general path uses a heap-allocated vector for arbitrary n.
    // The recv buffer always carries mpi_type so custom MPI types round-trip correctly.
    if (n == 1) {
        T staging_val{};
        kamping::v2::sendrecv(
            r | std::views::drop(local_size - 1) | kamping::v2::views::with_type(mpi_type),
            right,
            kamping::v2::views::ref_single(staging_val) | kamping::v2::views::with_type(mpi_type),
            left,
            comm
        );
        std::shift_right(first, last, 1);
        *first = std::move(staging_val);
    } else {
        std::vector<T> staging(static_cast<std::size_t>(n));
        kamping::v2::sendrecv(
            r | std::views::drop(local_size - n) | kamping::v2::views::with_type(mpi_type),
            right,
            staging | kamping::v2::views::with_type(mpi_type),
            left,
            comm
        );
        std::shift_right(first, last, n);
        std::ranges::copy(staging, first);
    }

    if (left == MPI_PROC_NULL) {
        // Rank 0: only the last (local_size - n) elements are from the global range.
        return {first + std::min(n, local_size), last};
    }
    return {first, last};
}

} // namespace dstl
