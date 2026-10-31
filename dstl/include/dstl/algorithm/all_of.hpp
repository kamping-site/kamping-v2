// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <functional>
#include <ranges>

#include <kamping/v2/collectives/allreduce.hpp>
#include <kamping/v2/views/ref_single_view.hpp>
#include <mpi.h>

#include "kamping/v2/handle.hpp"
#include "kamping/v2/sentinels.hpp"

namespace dstl {
template <kamping::v2::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
bool all_of(bool value, Comm const& comm = MPI_COMM_WORLD) {
    kamping::v2::allreduce(kamping::v2::inplace, kamping::v2::ref_single_view(value), std::logical_and<>{}, comm);
    return value;
}

template <
    std::input_iterator                              I,
    std::sentinel_for<I>                             S,
    kamping::v2::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm,
    class Proj                                            = std::identity,
    std::indirect_unary_predicate<std::projected<I, Proj>> Pred>
bool all_of(I first, S last, Pred pred, Comm const& comm = MPI_COMM_WORLD, Proj proj = {}) {
    bool all_of_local = std::ranges::all_of(first, last, std::ref(pred), std::ref(proj));
    return all_of(all_of_local, comm);
}
template <
    std::ranges::input_range                         R,
    kamping::v2::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm,
    class Proj                                            = std::identity,
    std::indirect_unary_predicate<std::projected<std::ranges::iterator_t<R>, Proj>> Pred>
bool all_of(R&& r, Pred pred, Comm const& comm = MPI_COMM_WORLD, Proj proj = {}) {
    return all_of(std::ranges::begin(r), std::ranges::end(r), std::ref(pred), comm, std::ref(proj));
}
} // namespace dstl
