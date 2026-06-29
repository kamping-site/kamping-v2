// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <ranges>
#include <vector>

#include <mpi.h>

#include "dstl/redistribute.hpp"
#include "kamping/v2/collectives/allgather.hpp"
#include "kamping/v2/collectives/allreduce.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/p2p/sendrecv.hpp"
#include "kamping/v2/sentinels.hpp"
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
/// For the regular case (n <= local size on every rank), each rank performs a blocking
/// sendrecv with its right neighbor to exchange the n boundary elements, then applies a
/// local std::shift_left.
///
/// For the irregular case (n exceeds the local size on at least one rank), the first n
/// elements are stripped from the global send distribution and dstl::redistribute is
/// called with the truncated send view and a correspondingly smaller recv count on the
/// trailing ranks. This ensures total send == total recv so redistribute operates
/// without holes.
///
/// @returns Per-rank subrange of r indicating the locally valid portion after the shift.
///   On ranks where all output elements originate from the original global array this is
///   [begin, end). On the rightmost affected rank(s) it covers only the elements with a
///   valid source; the remainder are value-initialized.
template <
    std::ranges::contiguous_range                           R,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
    requires mpi::experimental::send_recv_buffer<R>
std::ranges::borrowed_subrange_t<R>
shift_left(R&& r, std::ranges::range_difference_t<R> n, Comm const& comm = MPI_COMM_WORLD) {
    using T          = std::ranges::range_value_t<R>;
    using diff_t     = std::ranges::range_difference_t<R>;
    auto const first     = std::ranges::begin(r);
    auto const last      = std::ranges::end(r);
    diff_t const local_size = std::ranges::ssize(r);

    if (n <= 0) {
        return {first, last};
    }

    kamping::v2::comm_view cv{mpi::experimental::handle(comm)};
    int const rank = cv.rank();
    int const p    = cv.size();

    int const n_int = static_cast<int>(n);

    // Each rank decides locally whether the fast path is applicable for itself, then
    // a single-int allreduce (O(1) data, O(log p) latency) determines whether all
    // ranks can take it. This avoids the O(p)-data allgather on the common fast path.
    int use_fast_path = (n_int <= static_cast<int>(local_size)) ? 1 : 0;
    kamping::v2::allreduce(
        kamping::v2::inplace, kamping::v2::views::ref_single(use_fast_path), std::logical_and{}, cv);

    if (use_fast_path) {
        // Fast path: n does not exceed the local size on any rank.
        // For p==1 n==local_size==total is a no-op (matches std::shift_left semantics).
        // For p>1, n<=local_size implies n<total so no additional check is needed.
        if (p == 1 && n_int >= static_cast<int>(local_size)) {
            return {first, last};
        }

        // A single sendrecv with the right neighbor suffices.
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
                kamping::v2::views::ref_single(staging_val) |
                    kamping::v2::views::with_type(mpi_type),
                right,
                comm);
            std::shift_left(first, last, 1);
            *(first + (local_size - 1)) = std::move(staging_val);
        } else {
            std::vector<T> staging(static_cast<std::size_t>(n));
            kamping::v2::sendrecv(
                r | std::views::take(n) | kamping::v2::views::with_type(mpi_type),
                left,
                staging | kamping::v2::views::with_type(mpi_type),
                right,
                comm);
            std::shift_left(first, last, n);
            std::ranges::copy(staging, first + (local_size - n));
        }

        if (right == MPI_PROC_NULL) {
            return {first, first + std::max(diff_t{0}, local_size - n)};
        }
        return {first, last};
    }

    // General path: n exceeds the local size on at least one rank.
    //
    // Reformulated as redistribute: strip the first n global elements from the send
    // distribution. Each rank r drops max(0, n - offset[r]) elements from the front of
    // its local buffer (clamped to local_size). On the recv side each rank accepts only
    // the elements that fall within [0, total-n) of the output, so total send == total
    // recv. Trailing ranks get a smaller (or zero) recv count and zero-fill the tail.
    std::vector<int> sizes(static_cast<std::size_t>(p));
    sizes[static_cast<std::size_t>(rank)] = static_cast<int>(local_size);
    kamping::v2::allgather(kamping::v2::inplace, sizes, cv);

    int const my_offset = std::reduce(sizes.begin(), sizes.begin() + rank);
    int const total     = my_offset + std::reduce(sizes.begin() + rank, sizes.end());

    // n >= total: no effects (matches std::shift_left semantics).
    if (n_int >= total) {
        return {first, last};
    }

    MPI_Datatype const mpi_type = mpi::experimental::type(r);

    int const drop_count =
        std::min<int>(static_cast<int>(local_size), std::max(0, n_int - my_offset));
    int const valid_recv_count = std::max(
        0, std::min<int>(static_cast<int>(local_size), total - n_int - my_offset));

    // Copy the send portion into a staging buffer before receiving overwrites r.
    // Only the surviving suffix r[drop_count..] needs to be captured; the preceding
    // elements are discarded. Receiving directly into r avoids a second copy.
    std::vector<T> send_staging(first + drop_count, last);

    auto sbuf = send_staging | kamping::v2::views::with_type(mpi_type);
    auto rbuf = r | std::views::take(valid_recv_count) | kamping::v2::views::with_type(mpi_type);

    dstl::redistribute(sbuf, rbuf, comm);

    return {first, first + static_cast<diff_t>(valid_recv_count)};
}

/// @brief Shifts the distributed range right by n positions globally.
///
/// Distributed equivalent of std::shift_right applied to the concatenation of all local
/// ranges across the communicator. Elements at global position i are moved to global
/// position i + n. The first n elements of rank 0 are value-initialized.
///
/// For the regular case (n <= local size on every rank), each rank performs a blocking
/// sendrecv with its left neighbor to exchange the n boundary elements, then applies a
/// local std::shift_right.
///
/// For the irregular case, the last n elements are stripped from the global send
/// distribution and dstl::redistribute is called with the truncated send view and each
/// rank's recv buffer pointing into the non-zero suffix of the output staging vector.
/// The zero prefix is provided by value-initialization.
///
/// @returns Per-rank subrange of r indicating the locally valid portion after the shift.
///   On ranks where all output elements originate from the original global array this is
///   [begin, end). On the leftmost affected rank(s) it starts after begin(); the
///   preceding elements are value-initialized.
template <
    std::ranges::contiguous_range                           R,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
    requires mpi::experimental::send_recv_buffer<R>
std::ranges::borrowed_subrange_t<R>
shift_right(R&& r, std::ranges::range_difference_t<R> n, Comm const& comm = MPI_COMM_WORLD) {
    using T          = std::ranges::range_value_t<R>;
    using diff_t     = std::ranges::range_difference_t<R>;
    auto const first     = std::ranges::begin(r);
    auto const last      = std::ranges::end(r);
    diff_t const local_size = std::ranges::ssize(r);

    if (n <= 0) {
        return {first, last};
    }

    kamping::v2::comm_view cv{mpi::experimental::handle(comm)};
    int const rank = cv.rank();
    int const p    = cv.size();

    int const n_int = static_cast<int>(n);

    // Each rank decides locally whether the fast path is applicable for itself, then
    // a single-int allreduce (O(1) data, O(log p) latency) determines whether all
    // ranks can take it. This avoids the O(p)-data allgather on the common fast path.
    int use_fast_path = (n_int <= static_cast<int>(local_size)) ? 1 : 0;
    kamping::v2::allreduce(
        kamping::v2::inplace, kamping::v2::views::ref_single(use_fast_path), std::logical_and{}, cv);

    if (use_fast_path) {
        // Fast path: n does not exceed the local size on any rank.
        // For p==1 n==local_size==total is a no-op (matches std::shift_right semantics).
        // For p>1, n<=local_size implies n<total so no additional check is needed.
        if (p == 1 && n_int >= static_cast<int>(local_size)) {
            return {first, last};
        }

        // A single sendrecv with the left neighbor suffices.
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
                kamping::v2::views::ref_single(staging_val) |
                    kamping::v2::views::with_type(mpi_type),
                left,
                comm);
            std::shift_right(first, last, 1);
            *first = std::move(staging_val);
        } else {
            std::vector<T> staging(static_cast<std::size_t>(n));
            kamping::v2::sendrecv(
                r | std::views::drop(local_size - n) | kamping::v2::views::with_type(mpi_type),
                right,
                staging | kamping::v2::views::with_type(mpi_type),
                left,
                comm);
            std::shift_right(first, last, n);
            std::ranges::copy(staging, first);
        }

        if (left == MPI_PROC_NULL) {
            return {first + std::min(n, local_size), last};
        }
        return {first, last};
    }

    // General path: n exceeds the local size on at least one rank.
    //
    // Reformulated as redistribute: strip the last n global elements from the send
    // distribution. Each rank r keeps only max(0, min(local_size, total-n-offset[r]))
    // elements from the front of its buffer. On the recv side each rank writes into the
    // suffix output[valid_recv_start..local_size); the zero prefix [0, valid_recv_start)
    // is provided by value-initialization of the staging vector.
    std::vector<int> sizes(static_cast<std::size_t>(p));
    sizes[static_cast<std::size_t>(rank)] = static_cast<int>(local_size);
    kamping::v2::allgather(kamping::v2::inplace, sizes, cv);

    int const my_offset = std::reduce(sizes.begin(), sizes.begin() + rank);
    int const total     = my_offset + std::reduce(sizes.begin() + rank, sizes.end());

    // n >= total: no effects (matches std::shift_right semantics).
    if (n_int >= total) {
        return {first, last};
    }

    MPI_Datatype const mpi_type = mpi::experimental::type(r);

    int const keep_count = std::max(
        0, std::min<int>(static_cast<int>(local_size), total - n_int - my_offset));
    int const valid_recv_start = std::min<int>(
        static_cast<int>(local_size), std::max(0, n_int - my_offset));

    // Copy the send portion into a staging buffer before receiving overwrites r.
    // Only the surviving prefix r[0..keep_count) needs to be captured; the trailing
    // elements are discarded. Receiving directly into r avoids a second copy.
    std::vector<T> send_staging(first, first + keep_count);

    auto sbuf = send_staging | kamping::v2::views::with_type(mpi_type);
    auto rbuf = r | std::views::drop(valid_recv_start) | kamping::v2::views::with_type(mpi_type);

    dstl::redistribute(sbuf, rbuf, comm);

    return {first + static_cast<diff_t>(valid_recv_start), last};
}

} // namespace dstl
