// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <array>
#include <cstddef>
#include <list>
#include <map>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dstl/sparse_alltoall.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/p2p/mrecv.hpp"
#include "kamping/v2/p2p/probe.hpp"
#include "kamping/v2/views.hpp"

namespace {
template <typename T>
using sends_t = std::vector<std::pair<int, std::vector<T>>>;

/// All messages received by this process, keyed by source, in arrival order per source.
template <typename T>
using received_t = std::map<int, std::vector<std::vector<T>>>;

/// Returns a callback that receives every message into `received`.
template <typename T>
auto collect_into(received_t<T>& received) {
    return [&received](kamping::v2::probe_result probe) {
        int const      source = probe.source();
        std::vector<T> data;
        kamping::v2::mrecv(data | kamping::v2::views::resize, std::move(probe));
        received[source].push_back(std::move(data));
    };
}

/// Runs one exchange with `Algorithm` and returns what this process received.
template <typename Algorithm, typename Sends>
auto exchange(Sends const& sends, int tag = 0) {
    using T = kamping::v2::payload_element_t<Sends>; // works for both the sparse and the dense form
    received_t<T> received;
    dstl::sparse_alltoall(Algorithm{}, sends, collect_into(received), tag, MPI_COMM_WORLD);
    return received;
}

struct ranks {
    int rank;
    int size;
    int next;
    int prev;
};

ranks world_ranks() {
    kamping::v2::comm_view comm{MPI_COMM_WORLD};
    int const              rank = comm.rank();
    int const              p    = comm.size();
    return {rank, p, (rank + 1) % p, (rank - 1 + p) % p};
}

/// A buffer-protocol message that is not a std::ranges range (like a device buffer adapter).
struct int_buffer {
    using value_type = int;
    std::vector<int>         data;
    [[nodiscard]] int const* mpi_ptr() const {
        return data.data();
    }
    [[nodiscard]] std::size_t mpi_count() const {
        return data.size();
    }
};
} // namespace

/// Separates consecutive tests with a barrier: most tests use tag 0, and the documented contract requires consecutive
/// calls on the same communicator to use different tags or be synchronized. Without it, a fast process's next test
/// can send messages that a slow process still probing in the previous test matches.
class SynchronizedTest : public ::testing::Test {
protected:
    void TearDown() override {
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

template <typename Algorithm>
class SparseAlltoall : public SynchronizedTest {};

using SparseAlltoallRsx     = SynchronizedTest;
using SparseAlltoallDefault = SynchronizedTest;

using algorithms = ::testing::Types<
    dstl::sparse_alltoall_algorithm::nbx_t,
    dstl::sparse_alltoall_algorithm::pcx_t,
    dstl::sparse_alltoall_algorithm::rsx_t>;
TYPED_TEST_SUITE(SparseAlltoall, algorithms);

TYPED_TEST(SparseAlltoall, NoMessages) {
    EXPECT_TRUE(exchange<TypeParam>(sends_t<int>{}).empty());
}

TYPED_TEST(SparseAlltoall, EmptyMessagesAreNotSent) {
    auto const   r = world_ranks();
    sends_t<int> sends;
    for (int dest = 0; dest < r.size; ++dest) {
        sends.emplace_back(dest, std::vector<int>{});
    }
    EXPECT_TRUE(exchange<TypeParam>(sends).empty());
}

TYPED_TEST(SparseAlltoall, Ring) {
    auto const r        = world_ranks();
    auto       received = exchange<TypeParam>(sends_t<int>{{r.next, {r.rank, r.rank}}});
    ASSERT_EQ(received.size(), 1u);
    ASSERT_EQ(received[r.prev].size(), 1u);
    EXPECT_EQ(received[r.prev][0], (std::vector<int>{r.prev, r.prev}));
}

TYPED_TEST(SparseAlltoall, AllToAllWithVaryingSizes) {
    // Rank i sends to rank j a message of length j + 1 filled with i.
    auto const   r = world_ranks();
    sends_t<int> sends;
    for (int dest = 0; dest < r.size; ++dest) {
        sends.emplace_back(dest, std::vector<int>(static_cast<std::size_t>(dest + 1), r.rank));
    }
    auto received = exchange<TypeParam>(sends);

    ASSERT_EQ(received.size(), static_cast<std::size_t>(r.size));
    for (int source = 0; source < r.size; ++source) {
        ASSERT_EQ(received[source].size(), 1u);
        EXPECT_EQ(received[source][0], std::vector<int>(static_cast<std::size_t>(r.rank + 1), source));
    }
}

TYPED_TEST(SparseAlltoall, OnlySomeRanksSend) {
    // Only even ranks send, each to the next rank; odd ranks only receive (or nothing at all).
    auto const   r = world_ranks();
    sends_t<int> sends;
    if (r.rank % 2 == 0) {
        sends.emplace_back(r.next, std::vector<int>{r.rank});
    }
    auto received = exchange<TypeParam>(sends);

    bool const expects_message = r.prev % 2 == 0 && (r.size > 1 || r.rank == 0);
    if (expects_message) {
        ASSERT_EQ(received.size(), 1u);
        EXPECT_EQ(received[r.prev][0], (std::vector<int>{r.prev}));
    } else {
        EXPECT_TRUE(received.empty());
    }
}

TYPED_TEST(SparseAlltoall, SeveralMessagesToTheSameDestinationArriveInOrder) {
    auto const   r = world_ranks();
    sends_t<int> sends;
    for (int i = 0; i < 3; ++i) {
        sends.emplace_back(r.next, std::vector<int>{r.rank, i});
    }
    auto received = exchange<TypeParam>(sends);

    ASSERT_EQ(received.size(), 1u);
    ASSERT_EQ(received[r.prev].size(), 3u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(received[r.prev][static_cast<std::size_t>(i)], (std::vector<int>{r.prev, i}));
    }
}

TYPED_TEST(SparseAlltoall, LargeMessages) {
    // Large enough to exceed typical eager limits, so the rendezvous protocol is used.
    // Each rank s sends a message filled with s to its next rank, then one filled with -s to its previous rank.
    auto const        r        = world_ranks();
    std::size_t const n        = 1 << 18;
    auto              received = exchange<TypeParam>(
        sends_t<int>{{r.next, std::vector<int>(n, r.rank)}, {r.prev, std::vector<int>(n, -r.rank)}}
    );

    // With 1 or 2 ranks, next and previous coincide, so both messages can come from the same source (in send order).
    received_t<int> expected;
    for (int s = 0; s < r.size; ++s) {
        if ((s + 1) % r.size == r.rank) {
            expected[s].push_back(std::vector<int>(n, s));
        }
        if ((s - 1 + r.size) % r.size == r.rank) {
            expected[s].push_back(std::vector<int>(n, -s));
        }
    }
    EXPECT_EQ(received, expected);
}

TYPED_TEST(SparseAlltoall, DenseInputIndexedByDestination) {
    // Entry j is the message for rank j; only the next rank gets a non-empty one.
    auto const                    r = world_ranks();
    std::vector<std::vector<int>> sends(static_cast<std::size_t>(r.size));
    sends[static_cast<std::size_t>(r.next)] = {r.rank, 7};
    auto received                           = exchange<TypeParam>(sends);

    ASSERT_EQ(received.size(), 1u);
    ASSERT_EQ(received[r.prev].size(), 1u);
    EXPECT_EQ(received[r.prev][0], (std::vector<int>{r.prev, 7}));
}

TYPED_TEST(SparseAlltoall, DenseInputAllToAll) {
    // Rank i sends to rank j a message of length j + 1 filled with i, as in AllToAllWithVaryingSizes.
    auto const                    r = world_ranks();
    std::vector<std::vector<int>> sends;
    for (int dest = 0; dest < r.size; ++dest) {
        sends.emplace_back(static_cast<std::size_t>(dest + 1), r.rank);
    }
    auto received = exchange<TypeParam>(sends);

    ASSERT_EQ(received.size(), static_cast<std::size_t>(r.size));
    for (int source = 0; source < r.size; ++source) {
        ASSERT_EQ(received[source].size(), 1u);
        EXPECT_EQ(received[source][0], std::vector<int>(static_cast<std::size_t>(r.rank + 1), source));
    }
}

TYPED_TEST(SparseAlltoall, DenseInputShorterThanCommunicator) {
    // Fewer entries than processes: higher ranks simply get nothing.
    auto const                    r = world_ranks();
    std::vector<std::vector<int>> sends{{r.rank}}; // only to rank 0
    auto                          received = exchange<TypeParam>(sends);

    if (r.rank == 0) {
        ASSERT_EQ(received.size(), static_cast<std::size_t>(r.size));
        for (int source = 0; source < r.size; ++source) {
            EXPECT_EQ(received[source][0], std::vector<int>{source});
        }
    } else {
        EXPECT_TRUE(received.empty());
    }
}

TYPED_TEST(SparseAlltoall, BufferProtocolMessages) {
    // Messages that are not ranges, in the sparse and in the dense form.
    auto const r = world_ranks();

    std::vector<std::pair<int, int_buffer>> sparse{{r.next, int_buffer{{r.rank, 1}}}};
    auto                                    received = exchange<TypeParam>(sparse);
    ASSERT_EQ(received[r.prev].size(), 1u);
    EXPECT_EQ(received[r.prev][0], (std::vector<int>{r.prev, 1}));

    MPI_Barrier(MPI_COMM_WORLD); // second call with the same tag
    std::vector<int_buffer> dense(static_cast<std::size_t>(r.size));
    dense[static_cast<std::size_t>(r.next)].data = {r.rank, 2};
    received                                     = exchange<TypeParam>(dense);
    ASSERT_EQ(received[r.prev].size(), 1u);
    EXPECT_EQ(received[r.prev][0], (std::vector<int>{r.prev, 2}));
}

TYPED_TEST(SparseAlltoall, LazilyProducedNonOwningMessages) {
    // The input is a view producing (destination, span) pairs on the fly; the spans refer to `payloads`.
    auto const                    r = world_ranks();
    std::vector<std::vector<int>> payloads{{r.rank, 10}, {r.rank, 20}};
    auto                          sends    = payloads | std::views::transform([&](std::vector<int> const& payload) {
                     return std::pair{r.next, std::span<int const>(payload)};
                                             });
    auto                          received = exchange<TypeParam>(sends);

    ASSERT_EQ(received[r.prev].size(), 2u);
    EXPECT_EQ(received[r.prev][0], (std::vector<int>{r.prev, 10}));
    EXPECT_EQ(received[r.prev][1], (std::vector<int>{r.prev, 20}));
}

TYPED_TEST(SparseAlltoall, BuiltinValueTypeFallback) {
    auto const r        = world_ranks();
    auto       received = exchange<TypeParam>(sends_t<double>{{r.next, {0.5, double(r.rank)}}});
    ASSERT_EQ(received[r.prev].size(), 1u);
    EXPECT_EQ(received[r.prev][0], (std::vector<double>{0.5, double(r.prev)}));
}

TYPED_TEST(SparseAlltoall, ExplicitValueType) {
    // The value-type channel overrides the element type of the messages: send pairs of ints as MPI_2INT.
    auto const                                                    r = world_ranks();
    std::vector<std::pair<int, std::vector<std::pair<int, int>>>> sends{{r.next, {{r.rank, 1}, {r.rank, 2}}}};

    std::vector<std::pair<int, int>> received;
    int                              source = -1;
    dstl::sparse_alltoall(
        TypeParam{},
        sends | kamping::v2::views::with_value_type(MPI_2INT),
        [&](kamping::v2::probe_result probe) {
            source = probe.source();
            kamping::v2::mrecv(
                received | kamping::v2::views::with_type(MPI_2INT) | kamping::v2::views::resize,
                std::move(probe)
            );
        },
        0,
        MPI_COMM_WORLD
    );
    EXPECT_EQ(source, r.prev);
    EXPECT_EQ(received, (std::vector<std::pair<int, int>>{{r.prev, 1}, {r.prev, 2}}));
}

TYPED_TEST(SparseAlltoall, ConsecutiveCallsWithAlternatingTags) {
    // The documented isolation contract: consecutive calls use different tags. Each message carries its iteration.
    auto const r = world_ranks();
    for (int iteration = 0; iteration < 20; ++iteration) {
        auto received = exchange<TypeParam>(sends_t<int>{{r.next, {iteration, r.rank}}}, iteration % 2);
        ASSERT_EQ(received.size(), 1u) << "iteration " << iteration;
        ASSERT_EQ(received[r.prev].size(), 1u) << "iteration " << iteration;
        EXPECT_EQ(received[r.prev][0], (std::vector<int>{iteration, r.prev})) << "iteration " << iteration;
    }
}

TEST_F(SparseAlltoallRsx, CachedWindowIsReused) {
    auto const             r = world_ranks();
    dstl::rsx_window_cache windows;
    for (int iteration = 0; iteration < 5; ++iteration) {
        received_t<int> received;
        sends_t<int>    sends{{r.next, {iteration}}};
        if (iteration % 2 == 1) {
            sends.emplace_back(r.next, std::vector<int>{iteration, iteration}); // varying message counts
        }
        dstl::sparse_alltoall(
            dstl::sparse_alltoall_algorithm::rsx,
            sends,
            collect_into(received),
            iteration % 2,
            MPI_COMM_WORLD,
            windows
        );
        EXPECT_EQ(received[r.prev].size(), iteration % 2 == 1 ? 2u : 1u) << "iteration " << iteration;
        EXPECT_EQ(received[r.prev][0], std::vector<int>{iteration}) << "iteration " << iteration;
    }
    EXPECT_EQ(windows.size(), 1u);
}

TEST_F(SparseAlltoallDefault, WithoutAlgorithmTag) {
    auto const      r = world_ranks();
    received_t<int> received;
    dstl::sparse_alltoall(sends_t<int>{{r.next, {r.rank}}}, collect_into(received), 0, MPI_COMM_WORLD);
    ASSERT_EQ(received[r.prev].size(), 1u);
    EXPECT_EQ(received[r.prev][0], (std::vector<int>{r.prev}));
}

// ── Layer 2: receive into one buffer ─────────────────────────────────────────

namespace {
/// Rank s sends `messages_per_dest` messages to every rank d, each of length d + 1 filled with s.
sends_t<int> all_to_all_sends(int messages_per_dest = 1) {
    auto const   r = world_ranks();
    sends_t<int> sends;
    for (int dest = 0; dest < r.size; ++dest) {
        for (int m = 0; m < messages_per_dest; ++m) {
            sends.emplace_back(dest, std::vector<int>(static_cast<std::size_t>(dest + 1), r.rank));
        }
    }
    return sends;
}

/// Checks what `all_to_all_sends(messages_per_dest)` delivers to this rank: `out` starts with segments of length
/// rank + 1, each filled with its source, and every source appears in exactly `messages_per_dest` of them.
void expect_all_to_all_received(std::vector<int> const& out, std::size_t received, int messages_per_dest = 1) {
    auto const        r          = world_ranks();
    std::size_t const per_msg    = static_cast<std::size_t>(r.rank + 1);
    std::size_t const n_messages = static_cast<std::size_t>(r.size * messages_per_dest);
    ASSERT_EQ(received, n_messages * per_msg);
    ASSERT_GE(out.size(), received);
    std::map<int, int> per_source;
    for (std::size_t offset = 0; offset < received; offset += per_msg) {
        int const source = out[offset];
        for (std::size_t i = 0; i < per_msg; ++i) {
            EXPECT_EQ(out[offset + i], source) << "segment at " << offset;
        }
        ++per_source[source];
    }
    ASSERT_EQ(per_source.size(), static_cast<std::size_t>(r.size));
    for (auto const& [source, n]: per_source) {
        EXPECT_EQ(n, messages_per_dest) << "source " << source;
    }
}
} // namespace

template <typename Algorithm>
class SparseAlltoallIntoBuffer : public SynchronizedTest {};
TYPED_TEST_SUITE(SparseAlltoallIntoBuffer, algorithms);

TYPED_TEST(SparseAlltoallIntoBuffer, PresizedBuffer) {
    auto const       r = world_ranks();
    std::vector<int> out(static_cast<std::size_t>(r.size * (r.rank + 1)), -1);
    auto const       received = dstl::sparse_alltoall(TypeParam{}, all_to_all_sends(), out, 0, MPI_COMM_WORLD);
    expect_all_to_all_received(out, received);
}

TYPED_TEST(SparseAlltoallIntoBuffer, PresizedBufferWithManyMessages) {
    // Many receives in flight at once.
    auto const       r = world_ranks();
    std::vector<int> out(static_cast<std::size_t>(8 * r.size * (r.rank + 1)), -1);
    auto const       received = dstl::sparse_alltoall(TypeParam{}, all_to_all_sends(8), out, 0, MPI_COMM_WORLD);
    expect_all_to_all_received(out, received, 8);
}

TYPED_TEST(SparseAlltoallIntoBuffer, PresizedBufferLargerThanNeeded) {
    // Only the received elements are written; the rest of the buffer stays untouched.
    auto const        r      = world_ranks();
    std::size_t const needed = static_cast<std::size_t>(r.size * (r.rank + 1));
    std::vector<int>  out(needed + 3, -1);
    auto const        received = dstl::sparse_alltoall(TypeParam{}, all_to_all_sends(), out, 0, MPI_COMM_WORLD);
    EXPECT_EQ(received, needed);
    EXPECT_EQ(out.size(), needed + 3);
    EXPECT_EQ(out[needed], -1);
    EXPECT_EQ(out.back(), -1);
}

TYPED_TEST(SparseAlltoallIntoBuffer, ResizeBuffer) {
    std::vector<int> out{7, 7, 7}; // previous content is replaced
    auto const       received =
        dstl::sparse_alltoall(TypeParam{}, all_to_all_sends(), out | kamping::v2::views::resize, 0, MPI_COMM_WORLD);
    EXPECT_EQ(out.size(), received);
    expect_all_to_all_received(out, received);
}

TYPED_TEST(SparseAlltoallIntoBuffer, ResizeBufferWithManyMessages) {
    // Many messages per source, so the buffer grows many times while earlier messages must be kept.
    std::vector<int> out;
    auto const       received =
        dstl::sparse_alltoall(TypeParam{}, all_to_all_sends(8), out | kamping::v2::views::resize, 0, MPI_COMM_WORLD);
    EXPECT_EQ(out.size(), received);
    expect_all_to_all_received(out, received, 8);
}

TYPED_TEST(SparseAlltoallIntoBuffer, NoMessagesEmptiesAResizeBuffer) {
    std::vector<int> out{1, 2, 3};
    auto const       received =
        dstl::sparse_alltoall(TypeParam{}, sends_t<int>{}, out | kamping::v2::views::resize, 0, MPI_COMM_WORLD);
    EXPECT_EQ(received, 0u);
    EXPECT_TRUE(out.empty());
}

TYPED_TEST(SparseAlltoallIntoBuffer, FlatInputViaChunked) {
    // A flat buffer with per-rank counts, as for MPI_Alltoallv: rank s sends d + 1 copies of s to rank d.
    auto const       r = world_ranks();
    std::vector<int> counts;
    std::vector<int> data;
    for (int dest = 0; dest < r.size; ++dest) {
        counts.push_back(dest + 1);
        data.insert(data.end(), static_cast<std::size_t>(dest + 1), r.rank);
    }
    std::vector<int> out;
    auto const       received = dstl::sparse_alltoall(
        TypeParam{},
        data | kamping::v2::views::with_counts(counts) | kamping::v2::views::auto_displs()
            | kamping::v2::views::chunked,
        out | kamping::v2::views::resize,
        0,
        MPI_COMM_WORLD
    );
    expect_all_to_all_received(out, received);
}

TYPED_TEST(SparseAlltoallIntoBuffer, LargeMessages) {
    auto const        r = world_ranks();
    std::size_t const n = 1 << 18;
    std::vector<int>  out;
    auto const        received = dstl::sparse_alltoall(
        TypeParam{},
        sends_t<int>{{r.next, std::vector<int>(n, r.rank)}},
        out | kamping::v2::views::resize,
        0,
        MPI_COMM_WORLD
    );
    EXPECT_EQ(received, n);
    EXPECT_EQ(out, std::vector<int>(n, r.prev));
}

TEST_F(SparseAlltoallRsx, CachedWindowIntoBuffer) {
    dstl::rsx_window_cache windows;
    for (int iteration = 0; iteration < 3; ++iteration) {
        std::vector<int> out;
        auto const       received = dstl::sparse_alltoall(
            dstl::sparse_alltoall_algorithm::rsx,
            all_to_all_sends(),
            out | kamping::v2::views::resize,
            iteration % 2,
            MPI_COMM_WORLD,
            windows
        );
        expect_all_to_all_received(out, received);
    }
    EXPECT_EQ(windows.size(), 1u);
}

TEST_F(SparseAlltoallDefault, IntoBufferWithoutAlgorithmTag) {
    auto const       r = world_ranks();
    std::vector<int> out;
    auto const       received =
        dstl::sparse_alltoall(sends_t<int>{{r.next, {r.rank}}}, out | kamping::v2::views::resize, 0, MPI_COMM_WORLD);
    EXPECT_EQ(received, 1u);
    EXPECT_EQ(out, (std::vector<int>{r.prev}));
}

TEST(SparseAlltoallConcepts, RecvBuffer) {
    static_assert(dstl::sparse_alltoall_recv_buffer<std::vector<int>&>);
    static_assert(
        dstl::sparse_alltoall_recv_buffer<decltype(std::declval<std::vector<int>&>() | kamping::v2::views::resize)>
    );
    // A Layer 1 callback is never taken for a recv buffer.
    auto callback = [](kamping::v2::probe_result) {
    };
    static_assert(!dstl::sparse_alltoall_recv_buffer<decltype(callback)>);
}

TEST(SparseAlltoallConcepts, AlgorithmTags) {
    static_assert(dstl::is_sparse_alltoall_algorithm<dstl::sparse_alltoall_algorithm::nbx_t>);
    static_assert(dstl::is_sparse_alltoall_algorithm<dstl::sparse_alltoall_algorithm::pcx_t>);
    static_assert(dstl::is_sparse_alltoall_algorithm<dstl::sparse_alltoall_algorithm::rsx_t>);
    static_assert(!dstl::is_sparse_alltoall_algorithm<int>);
}

TEST(SparseAlltoallConcepts, SendBuffer) {
    // Sparse and dense forms, with contiguous messages.
    static_assert(dstl::sparse_alltoall_send_buffer<std::vector<std::pair<int, std::vector<int>>>>);
    static_assert(dstl::sparse_alltoall_send_buffer<std::vector<std::pair<int, std::span<int const>>>>);
    static_assert(dstl::sparse_alltoall_send_buffer<std::vector<std::vector<int>>>);
    static_assert(dstl::sparse_alltoall_send_buffer<std::vector<std::array<int, 2>>>); // dense, after the pair_like fix
    // Any buffer-protocol message, also ones that are not ranges.
    static_assert(dstl::sparse_alltoall_send_buffer<std::vector<int_buffer>>);
    static_assert(dstl::sparse_alltoall_send_buffer<std::vector<std::pair<int, int_buffer>>>);
    // Messages are sent in place, so ones that can only be iterated are rejected.
    static_assert(!dstl::sparse_alltoall_send_buffer<std::vector<std::pair<int, std::list<int>>>>);
    static_assert(!dstl::sparse_alltoall_send_buffer<std::vector<std::list<int>>>);
    // Flat buffers are not nested inputs.
    static_assert(!dstl::sparse_alltoall_send_buffer<std::vector<int>>);
}

namespace {
// Lazily transformed inputs, as types for the checks below.
std::vector<std::vector<int>> storage;
using owning_sparse   = decltype(storage | std::views::transform([](std::vector<int> const& v) {
                                   return std::pair{0, v}; // copies the message: owns its data
                                 }));
using borrowed_sparse = decltype(storage | std::views::transform([](std::vector<int> const& v) {
                                     return std::pair{0, std::span<int const>(v)}; // refers to storage
                                 }));
using owning_dense =
    decltype(storage | std::views::transform([](std::vector<int> const& v) { return std::vector<int>(v); }));
using borrowed_dense =
    decltype(storage | std::views::transform([](std::vector<int> const& v) { return std::span<int const>(v); }));
} // namespace

TEST(SparseAlltoallConcepts, MessagesOutliveIteration) {
    // Inputs that yield references: the messages live in storage that outlives the call.
    static_assert(dstl::messages_outlive_iteration<std::vector<std::pair<int, std::vector<int>>>>);
    static_assert(dstl::messages_outlive_iteration<std::vector<std::vector<int>>>);
    static_assert(dstl::messages_outlive_iteration<std::vector<int_buffer>>);
    // Messages produced by value: fine if they do not own their data ...
    static_assert(dstl::sparse_alltoall_send_buffer<borrowed_sparse>);
    static_assert(dstl::messages_outlive_iteration<borrowed_sparse>);
    static_assert(dstl::messages_outlive_iteration<borrowed_dense>);
    // ... rejected if they do: they would be destroyed while being sent.
    static_assert(dstl::sparse_alltoall_send_buffer<owning_sparse>);
    static_assert(!dstl::messages_outlive_iteration<owning_sparse>);
    static_assert(!dstl::messages_outlive_iteration<owning_dense>);
}
