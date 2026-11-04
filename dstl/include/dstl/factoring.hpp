// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <cstddef>
#include <vector>

#include <mpi.h>

#include "kamping/v2/comm.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "mpi/collectives/allreduce.hpp"
#include "mpi/comm.hpp"

/// @file
/// Pluggable factoring strategies (D4). A factoring strategy is a callable that,
/// given the global communicator, yields the factor vector {s_i} with
/// s_0 * s_1 * ... * s_{k-1} == p. The number of dimensions k is derived from the
/// returned vector. The default strategy is `dims_create{levels = 2}` (classic
/// 2-level grid, balanced via MPI_Dims_create).

namespace dstl {

/// A balanced grid with `levels` dimensions (default 2), computed with `MPI_Dims_create`: the dimensions multiply to
/// the number of processes and are as close to each other as possible.
struct dims_create {
    std::size_t levels = 2;

    std::vector<std::size_t> operator()(kamping::v2::comm_view global) const {
        auto             nnodes = global.size();
        auto             ndims  = static_cast<int>(levels);
        std::vector<int> dims(static_cast<std::size_t>(ndims), 0);
        MPI_Dims_create(nnodes, ndims, dims.data());
        return {dims.begin(), dims.end()};
    }
};

/// One dimension per hardware level: a two-dimensional grid of nodes × processes per node, derived from
/// `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`. Falls back to `dims_create{2}` if the nodes have different numbers of
/// processes.
struct topology_aware {
    std::vector<std::size_t> operator()(kamping::v2::comm_view global) const {
        auto     p      = static_cast<std::size_t>(global.size());
        MPI_Comm shared = MPI_COMM_NULL;
        MPI_Comm_split_type(global.mpi_handle(), MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared);
        int local_size = 0;
        MPI_Comm_size(shared, &local_size);
        MPI_Comm_free(&shared);
        // Require a homogeneous decomposition: every node must hold the same number of ranks.
        int min_size = 0;
        int max_size = 0;
        mpi::experimental::allreduce(
            kamping::v2::views::ref_single(local_size),
            kamping::v2::views::ref_single(min_size),
            MPI_MIN,
            global
        );
        mpi::experimental::allreduce(
            kamping::v2::views::ref_single(local_size),
            kamping::v2::views::ref_single(max_size),
            MPI_MAX,
            global
        );
        auto intra = static_cast<std::size_t>(local_size);
        if (min_size != max_size || intra == 0 || p % intra != 0) {
            return dims_create{2}(global);
        }
        std::size_t inter = p / intra;
        // Row-major convention: dimension 0 (largest stride, most remote) is the inter-node group;
        // the last dimension (stride 1, consecutive ranks) is the intra-node group.
        return {inter, intra};
    }
};

/// Dimensions given by the caller, used as they are; their product must equal the number of processes.
struct explicit_dims {
    std::vector<std::size_t> dims;

    std::vector<std::size_t> operator()(kamping::v2::comm_view /* global */) const {
        return dims;
    }
};

/// A factoring strategy: a callable that maps the global communicator to the grid dimensions, e.g. `dims_create`.
template <typename T>
concept factoring = requires(T const& strategy, kamping::v2::comm_view global) {
    { strategy(global) } -> std::convertible_to<std::vector<std::size_t>>;
};

} // namespace dstl
