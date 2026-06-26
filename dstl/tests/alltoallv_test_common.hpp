// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <tuple>
#include <vector>

#include <mpi.h>

#include "kamping/v2/collectives/alltoallv.hpp"
#include "kamping/v2/views.hpp"
#include "kamping/v2/views/auto_recv_v.hpp"

/// @file
/// Shared test fixtures for the dstl alltoallv suites (grid and flat): the world rank/size queries,
/// the send-data generator, the flat kamping::v2::alltoallv oracle, and a sort helper for multiset
/// comparisons. Both grid_alltoallv_test.cpp and flat_alltoallv_test.cpp include this so the two
/// suites validate against an identical setup and oracle.

namespace dstl_test {

inline int world_size() {
    int s = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
}

inline int world_rank() {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

/// Build a variadic send buffer: rank i sends (i+1) copies of (i*10 + j) to rank j.
/// @return {data, counts, displs} ready to pipe through views::with_counts / views::with_displs.
inline std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> build_send(int rank, int size) {
    std::vector<int> counts(static_cast<std::size_t>(size));
    std::vector<int> data;
    for (int j = 0; j < size; ++j) {
        counts[static_cast<std::size_t>(j)] = rank + 1;
        for (int k = 0; k < rank + 1; ++k) {
            data.push_back(rank * 10 + j);
        }
    }
    std::vector<int> displs(static_cast<std::size_t>(size));
    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
    return {data, counts, displs};
}

/// Oracle: the flat kamping::v2::alltoallv result on MPI_COMM_WORLD.
inline std::vector<int>
standard_alltoallv(std::vector<int> const& data, std::vector<int> const& counts, std::vector<int> const& displs) {
    namespace views = kamping::v2::views;
    std::vector<int> recv;
    kamping::v2::alltoallv(data | views::with_counts(counts) | views::with_displs(displs), recv | views::auto_recv_v);
    return recv;
}

/// Return a sorted copy, for multiset (order-independent) comparisons.
inline std::vector<int> sorted(std::vector<int> v) {
    std::ranges::sort(v);
    return v;
}

} // namespace dstl_test
