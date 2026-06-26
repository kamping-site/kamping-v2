// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <ranges>

#include <mpi/buffer.hpp>
#include <mpi/handle.hpp>

#include "dstl/algorithm/all_of.hpp"
#include "dstl/algorithm/shift.hpp"
#include "dstl/with_subset.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "kamping/v2/views/with_type_view.hpp"

namespace dstl {

/// @brief Tests whether the globally distributed range is sorted.
///
/// Distributed equivalent of std::ranges::is_sorted. Returns true iff:
///   1. Each rank's local slice is sorted under comp and proj, and
///   2. For every adjacent pair of non-empty ranks, the last element of the
///      left rank is not greater than the first element of the right rank.
///
/// Empty ranks are skipped via with_subset: the sub-communicator of non-empty
/// ranks checks their pair-wise boundaries with a single shift_right exchange.
template <
    std::ranges::forward_range                             R,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm                            = MPI_Comm,
    class Proj                                                                             = std::identity,
    std::indirect_strict_weak_order<std::projected<std::ranges::iterator_t<R>, Proj>> Comp = std::ranges::less>
    requires mpi::experimental::send_buffer<R>
bool is_sorted(R&& r, Comm const& comm = MPI_COMM_WORLD, Comp comp = {}, Proj proj = {}) {
    kamping::v2::comm_view const cv{mpi::experimental::handle(comm)};
    bool                         result = std::ranges::is_sorted(r, std::ref(comp), std::ref(proj));
    if (cv.size() == 1) {
        return result;
    }

    with_subset(cv, !std::ranges::empty(r), [&](kamping::v2::comm_view sub_comm) {
        using value_t = std::ranges::range_value_t<R>;

        // Put this rank's last element in a stack scalar; shift_right by 1
        // delivers it to the next rank's boundary via a zero-allocation sendrecv.
        MPI_Datatype const mpi_type = mpi::experimental::type(r);
        value_t boundary{std::ranges::data(r)[std::ranges::size(r) - 1]};
        shift_right(
            kamping::v2::views::ref_single(boundary) | kamping::v2::views::with_type(mpi_type),
            1,
            sub_comm
        );

        // Rank 0 of the sub-comm has no left neighbor; all others verify the
        // cross-rank pair: !comp(proj(first_this_rank), proj(last_prev_rank)).
        if (sub_comm.rank() > 0) {
            result = result
                  && !comp(std::invoke(proj, *std::ranges::begin(r)),
                            std::invoke(proj, boundary));
        }
    });

    return all_of(result, cv);
}

} // namespace dstl
