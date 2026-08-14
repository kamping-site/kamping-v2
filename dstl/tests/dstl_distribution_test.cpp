// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dstl/distribution.hpp"
#include "kamping/v2/comm.hpp"

// ── helpers ───────────────────────────────────────────────────────────────────

namespace {
// Triangular (ragged) per-rank size: rank r holds r+1 elements. Offsets and owners are
// derivable in closed form, which lets tests check every rank/index without relying on
// dstl::distribution itself for the expected values.
std::size_t triangular_local_size(int rank) {
    return static_cast<std::size_t>(rank) + 1;
}
std::size_t triangular_offset(int rank) {
    auto const r = static_cast<std::size_t>(rank);
    return r * (r + 1) / 2;
}
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// dstl::distribution
// ─────────────────────────────────────────────────────────────────────────────

TEST(Distribution, UniformSizes) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p       = comm.size();
    std::size_t const            local_n = 4;

    dstl::distribution const dist(local_n, comm);

    EXPECT_EQ(dist.size(), p);
    EXPECT_EQ(dist.global_size(), local_n * static_cast<std::size_t>(p));

    for (int r = 0; r < p; ++r) {
        EXPECT_EQ(dist.local_size(r), local_n);
        EXPECT_EQ(dist.index_range_begin(r), local_n * static_cast<std::size_t>(r));
        EXPECT_EQ(dist.index_range_end(r), local_n * static_cast<std::size_t>(r + 1));
    }
    for (std::size_t idx = 0; idx < dist.global_size(); ++idx) {
        EXPECT_EQ(dist.get_owner(idx), static_cast<int>(idx / local_n)) << "idx=" << idx;
    }

    auto const local  = dist.local_indices(comm.rank());
    auto const global = dist.global_indices(comm.rank());
    EXPECT_EQ(std::ranges::distance(local), static_cast<std::ptrdiff_t>(local_n));
    EXPECT_EQ(*local.begin(), 0u);
    EXPECT_EQ(*global.begin(), dist.index_range_begin(comm.rank()));
}

TEST(Distribution, RaggedSizes) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p = comm.size();

    dstl::distribution const dist(triangular_local_size(comm.rank()), comm);

    EXPECT_EQ(dist.global_size(), triangular_offset(p));

    for (int r = 0; r < p; ++r) {
        EXPECT_EQ(dist.local_size(r), triangular_local_size(r));
        EXPECT_EQ(dist.index_range_begin(r), triangular_offset(r));
        EXPECT_EQ(dist.index_range_end(r), triangular_offset(r + 1));

        // owner at both boundaries of rank r's window
        EXPECT_EQ(dist.get_owner(dist.index_range_begin(r)), r);
        EXPECT_EQ(dist.get_owner(dist.index_range_end(r) - 1), r);
    }
}

TEST(Distribution, ExplicitOffsetsConstructor) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p = comm.size();

    std::vector<std::size_t> offsets;
    offsets.reserve(static_cast<std::size_t>(p) + 1);
    for (int r = 0; r <= p; ++r) {
        offsets.push_back(triangular_offset(r));
    }

    dstl::distribution const explicit_dist(offsets, comm);
    dstl::distribution const gathered_dist(triangular_local_size(comm.rank()), comm);

    EXPECT_EQ(explicit_dist.global_size(), gathered_dist.global_size());
    for (int r = 0; r < p; ++r) {
        EXPECT_EQ(explicit_dist.local_size(r), gathered_dist.local_size(r));
        EXPECT_EQ(explicit_dist.index_range_begin(r), gathered_dist.index_range_begin(r));
    }
    for (std::size_t idx = 0; idx < explicit_dist.global_size(); ++idx) {
        EXPECT_EQ(explicit_dist.get_owner(idx), gathered_dist.get_owner(idx)) << "idx=" << idx;
    }
}

TEST(Distribution, SingleRank) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    if (comm.size() != 1) {
        GTEST_SKIP() << "single-rank only";
    }
    dstl::distribution const dist(5, comm);

    EXPECT_EQ(dist.size(), 1);
    EXPECT_EQ(dist.global_size(), 5u);
    EXPECT_EQ(dist.local_size(0), 5u);
    for (std::size_t idx = 0; idx < 5; ++idx) {
        EXPECT_EQ(dist.get_owner(idx), 0);
    }
}

TEST(Distribution, LocalGlobalRoundTrip) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p = comm.size();

    dstl::distribution const dist(triangular_local_size(comm.rank()), comm);

    for (int r = 0; r < p; ++r) {
        for (std::size_t i = 0; i < dist.local_size(r); ++i) {
            auto const global_idx = dist.local_to_global(i, r);
            EXPECT_TRUE(dist.is_local(global_idx, r));
            EXPECT_EQ(dist.global_to_local(global_idx, r), i);
        }
    }
}

TEST(Distribution, OwnerAtPartitionBoundaries) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p = comm.size();

    // Uniform distribution.
    {
        dstl::distribution const dist(4, comm);
        for (int r = 0; r < p; ++r) {
            EXPECT_EQ(dist.get_owner(dist.index_range_end(r) - 1), r);
            if (r + 1 < p) {
                EXPECT_EQ(dist.get_owner(dist.index_range_end(r)), r + 1);
            }
        }
    }
    // Ragged distribution.
    {
        dstl::distribution const dist(triangular_local_size(comm.rank()), comm);
        for (int r = 0; r < p; ++r) {
            EXPECT_EQ(dist.get_owner(dist.index_range_end(r) - 1), r);
            if (r + 1 < p) {
                EXPECT_EQ(dist.get_owner(dist.index_range_end(r)), r + 1);
            }
        }
    }
}

TEST(Distribution, EmptyLocalRangeOnSomeRanks) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p       = comm.size();
    std::size_t const            local_n = (comm.rank() == 0) ? 0 : 3;

    dstl::distribution const dist(local_n, comm);

    EXPECT_EQ(dist.local_size(0), 0u);
    EXPECT_TRUE(dist.local_indices(0).empty());
    EXPECT_EQ(dist.index_range_begin(0), dist.index_range_end(0));

    for (std::size_t idx = 0; idx < dist.global_size(); ++idx) {
        EXPECT_NE(dist.get_owner(idx), 0) << "idx=" << idx << " must not resolve to the empty rank 0";
    }
    if (p > 1) {
        EXPECT_EQ(dist.get_owner(0), 1);
    }
}

TEST(Distribution, Counts) {
    kamping::v2::comm_view const comm{MPI_COMM_WORLD};
    int const                    p = comm.size();

    dstl::distribution const dist(triangular_local_size(comm.rank()), comm);

    auto const               counts_view = dist.counts();
    std::vector<std::size_t> counts(counts_view.begin(), counts_view.end());
    ASSERT_EQ(counts.size(), static_cast<std::size_t>(p));
    for (int r = 0; r < p; ++r) {
        EXPECT_EQ(counts[static_cast<std::size_t>(r)], triangular_local_size(r));
        EXPECT_EQ(counts[static_cast<std::size_t>(r)], dist.local_size(r));
    }
}
