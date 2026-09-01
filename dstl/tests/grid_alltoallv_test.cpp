// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "alltoallv_test_common.hpp"
#include "dstl/dstl.hpp"
#include "kamping/v2/views.hpp"
#include "mpi/comm.hpp"

using namespace ::testing;
using mpi::experimental::comm_view;
namespace views = kamping::v2::views;
using dstl_test::build_send;
using dstl_test::sorted;
using dstl_test::standard_alltoallv;
using dstl_test::world_rank;
using dstl_test::world_size;

// unordered: the recv buffer is multiset-equal to the flat alltoallv.
TEST(GridAlltoallvTest, UnorderedMultisetEqualsFlat) {
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);

    std::vector<int> expected = standard_alltoallv(data, counts, displs);

    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::grid_alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv | views::resize, // opt into automatic resizing (bare buffers are assumed pre-sized)
        grid,
        dstl::layout::unordered{}
    );

    EXPECT_EQ(sorted(recv), sorted(expected));
}

// ordered_by_source: the recv buffer is element-identical to the flat alltoallv.
TEST(GridAlltoallvTest, OrderedEqualsFlatExactly) {
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);

    std::vector<int> expected = standard_alltoallv(data, counts, displs);

    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::grid_alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv | views::auto_recv_v,
        grid,
        dstl::layout::ordered_by_source{}
    );

    EXPECT_EQ(recv, expected);
}

// Owned (rvalue) recv buffer: the data lives in the returned result.
TEST(GridAlltoallvTest, OwnedRecvBuffer) {
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);

    std::vector<int> expected = standard_alltoallv(data, counts, displs);

    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    auto                       res = dstl::grid_alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        std::vector<int>{} | views::auto_recv_v,
        grid,
        dstl::layout::ordered_by_source{}
    );

    EXPECT_EQ(res.recv.underlying(), expected);
}

// Each rank sends exactly one element to each rank — same multiset as alltoall.
TEST(GridAlltoallvTest, UniformSingleElement) {
    int              rank = world_rank();
    int              size = world_size();
    std::vector<int> data(static_cast<std::size_t>(size), rank);
    std::vector<int> counts(static_cast<std::size_t>(size), 1);
    std::vector<int> displs(static_cast<std::size_t>(size));
    std::iota(displs.begin(), displs.end(), 0);

    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::grid_alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv | views::auto_recv_v,
        grid,
        dstl::layout::ordered_by_source{}
    );

    std::vector<int> expected(static_cast<std::size_t>(size));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(recv, expected);
}

// Degenerate: every rank sends nothing.
TEST(GridAlltoallvTest, AllEmpty) {
    int                        size = world_size();
    std::vector<int>           data;
    std::vector<int>           counts(static_cast<std::size_t>(size), 0);
    std::vector<int>           displs(static_cast<std::size_t>(size), 0);
    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::grid_alltoallv(
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
        recv,
        grid
    );
    EXPECT_TRUE(recv.empty());
}

// Opt-in resize: a PRE-SIZED bare recv buffer (no views::resize) is written into as-is.
TEST(GridAlltoallvTest, PreSizedBareRecvBuffer) {
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);

    std::vector<int> expected = standard_alltoallv(data, counts, displs);

    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv(expected.size()); // caller pre-sizes; no views::resize
    dstl::grid_alltoallv(data | views::with_counts(counts) | views::with_displs(displs), recv, grid, dstl::layout::unordered{});

    EXPECT_EQ(sorted(recv), sorted(expected));
}

// Headline of the relaxed contract: send and recv may use DIFFERENT element types / datatypes as long as
// the MPI type signatures match (exactly like a plain MPI_Alltoallv). Send a struct with a gap whose
// datatype spans only two of three members; receive into a packed two-int struct. The gap is never
// transmitted; the two payload ints arrive intact.
TEST(GridAlltoallvTest, MixedGappedSendPackedRecv) {
    int rank = world_rank();
    int size = world_size();

    struct SendS {
        int a;
        int gap;
        int b;
    };
    struct RecvS {
        int a;
        int b;
    };

    // dt_send: blocks {a@offsetof(a), b@offsetof(b)} of MPI_INT, extent resized to sizeof(SendS).
    MPI_Datatype dt_send;
    {
        int          blocklen[2] = {1, 1};
        MPI_Aint     disp[2]     = {offsetof(SendS, a), offsetof(SendS, b)};
        MPI_Datatype types[2]    = {MPI_INT, MPI_INT};
        MPI_Datatype tmp;
        MPI_Type_create_struct(2, blocklen, disp, types, &tmp);
        MPI_Type_create_resized(tmp, 0, static_cast<MPI_Aint>(sizeof(SendS)), &dt_send);
        MPI_Type_commit(&dt_send);
        MPI_Type_free(&tmp);
    }
    // dt_recv: two contiguous ints, extent sizeof(RecvS). Same signature (INT, INT) as dt_send.
    MPI_Datatype dt_recv;
    MPI_Type_contiguous(2, MPI_INT, &dt_recv);
    MPI_Type_commit(&dt_recv);

    // Each rank sends exactly one SendS to every rank d: a = rank*100 + d, b = rank*1000 + d, gap = -1.
    std::vector<SendS> data(static_cast<std::size_t>(size));
    for (int d = 0; d < size; ++d) {
        data[static_cast<std::size_t>(d)] = SendS{rank * 100 + d, -1, rank * 1000 + d};
    }
    std::vector<int> counts(static_cast<std::size_t>(size), 1);
    std::vector<int> displs(static_cast<std::size_t>(size));
    std::iota(displs.begin(), displs.end(), 0);

    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<RecvS>         recv;
    dstl::grid_alltoallv(
        data | views::with_type(dt_send) | views::with_counts(counts) | views::with_displs(displs),
        recv | views::with_type(dt_recv) | views::resize,
        grid,
        dstl::layout::unordered{}
    );

    // Rank d receives one element from each source r: (a, b) = (r*100 + d, r*1000 + d), gap dropped.
    std::vector<std::pair<int, int>> got;
    for (auto const& e: recv) {
        got.emplace_back(e.a, e.b);
    }
    std::vector<std::pair<int, int>> want;
    for (int r = 0; r < size; ++r) {
        want.emplace_back(r * 100 + rank, r * 1000 + rank);
    }
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    EXPECT_EQ(got, want);

    MPI_Type_free(&dt_send);
    MPI_Type_free(&dt_recv);
}

// Deferred send buffer: sparse flatten_v() sends require set_comm_size() before counts/displs are
// read. Without the fix, ensure_flattened() would assert comm_size_.has_value() and abort.
TEST(GridAlltoallvTest, DeferredSendBuf_SparseFlattenV) {
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);
    std::vector<int> expected   = standard_alltoallv(data, counts, displs);

    // Same send pattern as build_send but packaged as sparse (destination, buffer) pairs in
    // reverse rank order — this triggers the sparse path in flatten_v_view that requires
    // set_comm_size() before it can lay out counts/displs/data.
    std::vector<std::pair<int, std::vector<int>>> per_dest;
    for (int j = size - 1; j >= 0; --j) {
        per_dest.emplace_back(j, std::vector<int>(static_cast<std::size_t>(rank + 1), rank * 10 + j));
    }

    dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};
    std::vector<int>           recv;
    dstl::grid_alltoallv(per_dest | views::flatten_v(), recv | views::resize, grid, dstl::layout::unordered{});

    EXPECT_EQ(sorted(recv), sorted(expected));
}

// Explicit non-default factorizations produce the same result (exercises different k / dims).
TEST(GridAlltoallvTest, ExplicitFactorizations) {
    int rank                    = world_rank();
    int size                    = world_size();
    auto [data, counts, displs] = build_send(rank, size);
    std::vector<int> expected   = standard_alltoallv(data, counts, displs);

    // A 1-D grid (flat), and — when 4 ranks — a 2x2 grid.
    std::vector<std::vector<std::size_t>> factorings = {{static_cast<std::size_t>(size)}};
    if (size == 4) {
        factorings.push_back({2, 2});
    }
    if (size == 8) {
        factorings.push_back({2, 4});
        factorings.push_back({2, 2, 2});
    }

    for (auto const& dims: factorings) {
        dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}, std::span<std::size_t const>{dims}};
        std::vector<int>           recv;
        dstl::grid_alltoallv(
            data | kamping::v2::views::with_counts(counts) | kamping::v2::views::with_displs(displs),
            recv | views::auto_recv_v,
            grid,
            dstl::layout::ordered_by_source{}
        );
        EXPECT_EQ(recv, expected) << "factorization with k=" << dims.size();
    }
}

// Isolated reproducer for the SuperMUC/Intel-MPI grid_alltoallv bug tracked on KaCCv2's
// feat/frozen-multistep branch (see that repo's notes/grid-alltoallv-supermuc-data-
// loss.md and git log for the full investigation). Replays the exact per-round
// send-count sequence a real SuperMUC repro produced (FrozenMultiStep's kout sampling
// strategy, sync-grid reachability search, gnm-undirected n=524288/m=4194304, p=2),
// over a dstl::grid_comm rebuilt fresh every "iteration" -- exactly like KaCCv2's
// find_big_component_multi does once per benchmark iteration -- with an element type
// shaped like KaCCv2's actual wire Message (a uint64_t vertex id + an int payload,
// MPI_Type_create_struct+resized to 16 bytes -- same derived-datatype construction
// MixedGappedSendPackedRecv above already exercises, reused here instead of guessing at
// how kamping-v2 bridges to kamping v1's std::pair<uint64_t,int> type registration that
// KaCCv2's bfs.hpp actually relies on).
//
// Poison-fills each round's receive buffer with 0xAA before the real
// dstl::grid_alltoallv call and reports (to stderr, plus a hard test failure) any
// element still showing the poison pattern afterward -- i.e. an element MPI claimed to
// deliver but never actually wrote. On the real repro this was proven NOT to be
// corrupted content: every element that failed the receiving rank's is_local() check
// decoded to exactly the 0xAA pattern, on both ranks, confirmed via this same
// poison-fill technique added directly to KaCCv2's production code path.
//
// Needs genuine multi-node execution (2 real SuperMUC nodes, 1 rank/node,
// `mpiexec -n 2 --perhost 1`). Set KACC_REPRO_ITERATIONS (default 5, matching KaCCv2's
// own `--iterations 5`) to change how many times the round sequence repeats -- the real
// bug has needed 2-3 repeats (not 1) to manifest, and the exact number has varied run to
// run.
//
// The plain version below (SizeTransitionPoisonRepro) ran on real SuperMUC hardware
// (2 nodes, Intel MPI 2021.17.0) FOUR separate times and never reproduced -- despite
// replaying the exact size sequence that reliably crashes real KaCC. The isolated
// sequence completes in under a second; the real benchmark takes 17-33s per config,
// dominated by real computation (neighbor scanning over hundreds of thousands of
// elements) between rounds that this synthetic version skips entirely, and by a large,
// persistent graph data structure resident in memory that this synthetic version never
// allocates. The three variants below each add back one candidate factor, in isolation,
// to tell them apart: real elapsed time between calls, real background memory pressure,
// or both together. All four tests match --gtest_filter='*SizeTransitionPoisonRepro*',
// so the existing sbatch script runs all of them in one job with no changes needed.
namespace grid_alltoallv_repro {

struct Message {
    std::uint64_t vid;
    int           payload;
};

MPI_Datatype message_datatype() {
    int          blocklen[2] = {1, 1};
    MPI_Aint     disp[2]     = {offsetof(Message, vid), offsetof(Message, payload)};
    MPI_Datatype types[2]    = {MPI_UINT64_T, MPI_INT};
    MPI_Datatype tmp;
    MPI_Type_create_struct(2, blocklen, disp, types, &tmp);
    MPI_Datatype dt;
    MPI_Type_create_resized(tmp, 0, static_cast<MPI_Aint>(sizeof(Message)), &dt);
    MPI_Type_commit(&dt);
    MPI_Type_free(&tmp);
    return dt;
}

// {rank0->1, rank1->0} per round, read directly off a real SuperMUC repro's route_phase
// trace. Deterministic across iterations there (fixed seed), so this exact sequence
// repeating below is a faithful replay, not an approximation.
struct RoundCounts {
    std::size_t to_1;
    std::size_t to_0;
};
constexpr RoundCounts kRounds[] = {
    {396252, 396779},
    {111004, 109734},
    {6436, 6404},
    {322, 335},
    {8, 23},
    {0, 5},
    {0, 0},
};

// Real elapsed time (ms) spent BEFORE each round's exchange completes, read off the
// same real SuperMUC trace's timestamps -- dominated by the real neighbor-scan/
// next_frontier-build work between rounds, which this synthetic reproducer otherwise
// skips (the whole 7-round sequence runs in well under a second here vs. several
// seconds for just the first two rounds on real hardware).
constexpr int kRoundDelayMs[] = {0, 1240, 150, 20, 5, 2, 1};

// Roughly this repro's graph's real per-rank memory footprint order of magnitude
// (m=4194304 edges * 8-byte VId * ~2 for undirected + ghost/offset overhead, rounded
// up) -- NOT an exact figure, just calibrated to the right ballpark.
constexpr std::size_t kResidentBytes = 200ULL * 1024 * 1024;

int repro_iterations() {
    if (char const* env = std::getenv("KACC_REPRO_ITERATIONS")) {
        return std::max(1, std::atoi(env));
    }
    return 5;
}

// Runs the full round sequence `repro_iterations()` times, poison-filling before each
// real alltoallv call. `before_round(iter, round)` fires immediately before each
// round's exchange -- e.g. to sleep (timing hypothesis) or touch a large persistent
// allocation (memory-pressure hypothesis). Returns the total poison count found,
// reporting per-round detail to stderr as it goes.
template <typename BeforeRound>
std::size_t run(BeforeRound&& before_round) {
    int const    rank  = world_rank();
    int const    other = 1 - rank;
    MPI_Datatype dt    = message_datatype();

    std::size_t total_poison = 0;
    for (int iter = 1; iter <= repro_iterations(); ++iter) {
        dstl::grid_comm<dstl::execution_policy::seq> grid{comm_view{MPI_COMM_WORLD}};

        for (std::size_t round = 0; round < std::size(kRounds); ++round) {
            before_round(iter, static_cast<int>(round));

            std::size_t const send_count = rank == 0 ? kRounds[round].to_1 : kRounds[round].to_0;
            std::size_t const recv_count = rank == 0 ? kRounds[round].to_0 : kRounds[round].to_1;

            std::vector<int> send_counts(2, 0);
            std::vector<int> send_displs(2, 0);
            send_counts[static_cast<std::size_t>(other)] = static_cast<int>(send_count);

            std::vector<Message> send_data(send_count);
            for (std::size_t i = 0; i < send_count; ++i) {
                send_data[i] = Message{
                    static_cast<std::uint64_t>(iter) * 100000000ULL
                        + static_cast<std::uint64_t>(round) * 1000000ULL + i,
                    rank
                };
            }

            std::vector<Message> recv_data(recv_count);
            std::memset(recv_data.data(), 0xAA, recv_data.size() * sizeof(Message));

            dstl::grid_alltoallv(
                send_data | views::with_type(dt) | views::with_counts(send_counts) | views::with_displs(send_displs),
                recv_data | views::with_type(dt),
                grid
            );

            std::size_t poison = 0;
            for (auto const& elem: recv_data) {
                if (elem.vid == 0xAAAAAAAAAAAAAAAAULL && elem.payload == static_cast<int>(0xAAAAAAAA)) {
                    ++poison;
                }
            }
            if (poison > 0) {
                std::fprintf(
                    stderr,
                    "[repro] *** REPRODUCED *** iter=%d round=%zu rank=%d: %zu/%zu elements never written by "
                    "MPI_Alltoallv (still show the 0xAA poison pattern)\n",
                    iter,
                    round,
                    rank,
                    poison,
                    recv_data.size()
                );
                std::fflush(stderr);
            }
            total_poison += poison;
        }
    }

    MPI_Type_free(&dt);
    return total_poison;
}

// Allocates and touches (writes every byte of, forcing real page commit -- reserve/
// resize alone doesn't) a large block once per iteration, refreshed each time a new
// iteration starts -- mimicking a graph ingested once and kept resident for the whole
// iteration's rounds, rather than truly-static memory the allocator would just keep
// handing back the same already-committed pages for. `resident` and `last_iter` are the
// caller's, so the allocation survives across `before_round` calls within one test.
void touch_resident_block(std::optional<std::vector<unsigned char>>& resident, int& last_iter, int iter) {
    if (iter == last_iter) {
        return;
    }
    last_iter = iter;
    resident.emplace(kResidentBytes);
    std::fill(resident->begin(), resident->end(), static_cast<unsigned char>(iter));
}

} // namespace grid_alltoallv_repro

TEST(GridAlltoallvTest, SizeTransitionPoisonRepro) {
    if (world_size() != 2) {
        GTEST_SKIP() << "calibrated for exactly 2 ranks (the real repro's p)";
    }
    using namespace grid_alltoallv_repro;
    std::size_t total_poison = run([](int, int) {});
    EXPECT_EQ(total_poison, 0u) << "at least one grid_alltoallv call left part of its receive buffer "
                                    "unwritten -- see stderr for which iteration/round/rank";
}

// Timing hypothesis: does this need real elapsed wall-clock time between calls (the
// synthetic sequence otherwise fires all 7 rounds in well under a second), separate
// from any real computation or memory effect?
TEST(GridAlltoallvTest, SizeTransitionPoisonReproWithDelay) {
    if (world_size() != 2) {
        GTEST_SKIP() << "calibrated for exactly 2 ranks (the real repro's p)";
    }
    using namespace grid_alltoallv_repro;
    std::size_t total_poison = run([](int, int round) {
        usleep(static_cast<useconds_t>(kRoundDelayMs[static_cast<std::size_t>(round)]) * 1000);
    });
    EXPECT_EQ(total_poison, 0u) << "at least one grid_alltoallv call left part of its receive buffer "
                                    "unwritten -- see stderr for which iteration/round/rank";
}

// Memory-pressure hypothesis: does this need a large, persistent background allocation
// coexisting with the per-round send/recv buffers (mimicking the real graph's resident
// memory), separate from any timing effect?
TEST(GridAlltoallvTest, SizeTransitionPoisonReproWithMemoryPressure) {
    if (world_size() != 2) {
        GTEST_SKIP() << "calibrated for exactly 2 ranks (the real repro's p)";
    }
    using namespace grid_alltoallv_repro;
    std::optional<std::vector<unsigned char>> resident;
    int                                        last_iter = 0;
    std::size_t total_poison = run([&](int iter, int round) {
        (void)round;
        touch_resident_block(resident, last_iter, iter);
    });
    EXPECT_EQ(total_poison, 0u) << "at least one grid_alltoallv call left part of its receive buffer "
                                    "unwritten -- see stderr for which iteration/round/rank";
}

// Combined: both factors together, in case neither alone is sufficient.
TEST(GridAlltoallvTest, SizeTransitionPoisonReproWithDelayAndMemoryPressure) {
    if (world_size() != 2) {
        GTEST_SKIP() << "calibrated for exactly 2 ranks (the real repro's p)";
    }
    using namespace grid_alltoallv_repro;
    std::optional<std::vector<unsigned char>> resident;
    int                                        last_iter = 0;
    std::size_t total_poison = run([&](int iter, int round) {
        touch_resident_block(resident, last_iter, iter);
        usleep(static_cast<useconds_t>(kRoundDelayMs[static_cast<std::size_t>(round)]) * 1000);
    });
    EXPECT_EQ(total_poison, 0u) << "at least one grid_alltoallv call left part of its receive buffer "
                                    "unwritten -- see stderr for which iteration/round/rank";
}
