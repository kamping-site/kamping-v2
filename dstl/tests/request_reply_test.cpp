// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dstl/dstl.hpp"
#include "kamping/v2/views.hpp"
#include "mpi/comm.hpp"

namespace views = kamping::v2::views;
using mpi::experimental::comm_view;

namespace {

int world_size() {
    return comm_view{MPI_COMM_WORLD}.size();
}

int world_rank() {
    return comm_view{MPI_COMM_WORLD}.rank();
}

/// An interleaved (not destination-grouped) request set: rank r issues `3*size + r` requests, the k-th
/// addressed to rank (r + k) % size, so the internal bucketing must actually permute.
std::vector<std::pair<int, int>> build_pairs(int rank, int size) {
    std::vector<std::pair<int, int>> reqs;
    int const                        total = 3 * size + rank;
    for (int k = 0; k < total; ++k) {
        int const target = (rank + k) % size;
        int const value  = rank * 1000 + k;
        reqs.emplace_back(value, target);
    }
    return reqs;
}

/// Oracle for the responder-grouped output: replies laid out in per-responder blocks (block r at
/// exclusive-scan offset), each block holding the replies to this rank's requests bound for r in input
/// order — exactly the layout the reverse exchange deposits.
template <typename Reply, typename ReplyOf>
std::vector<Reply>
expected_grouped(std::vector<std::pair<int, int>> const& reqs, int size, ReplyOf reply_of) {
    std::vector<int> counts(static_cast<std::size_t>(size), 0);
    for (auto const& [v, d]: reqs) {
        ++counts[static_cast<std::size_t>(d)];
    }
    std::vector<int> cursor(static_cast<std::size_t>(size));
    std::exclusive_scan(counts.begin(), counts.end(), cursor.begin(), 0);
    std::vector<Reply> out(reqs.size());
    for (auto const& [v, d]: reqs) {
        out[static_cast<std::size_t>(cursor[static_cast<std::size_t>(d)]++)] = reply_of(v);
    }
    return out;
}

std::vector<int> sorted(std::vector<int> v) {
    std::ranges::sort(v);
    return v;
}

} // namespace

// ── exchange_layout / reverse_layout ────────────────────────────────────────────────────────────────

// A forward alltoallv, then a reverse alltoallv driven by the reversed layout, is position preserving: a
// reply written for received element j comes back to the exact slot of the element that produced it, with
// no count negotiation on the return.
TEST(RequestReplyExchange, ReverseLayoutRoundTrip) {
    int              rank = world_rank();
    int              size = world_size();
    std::vector<int> data;
    std::vector<int> counts(static_cast<std::size_t>(size), rank + 1);
    for (int j = 0; j < size; ++j) {
        for (int k = 0; k < rank + 1; ++k) {
            data.push_back(rank * 100 + j);
        }
    }
    std::vector<int> displs(static_cast<std::size_t>(size));
    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);

    // Forward, then snapshot the negotiated layout off the two buffers.
    std::vector<int> recv_requests;
    auto             fwd =
        kamping::v2::alltoallv(data | views::with_counts(counts) | views::with_displs(displs), recv_requests | views::auto_recv_v);
    dstl::exchange_layout layout{fwd.send, fwd.recv};

    std::vector<int> replies(recv_requests.size());
    for (std::size_t j = 0; j < recv_requests.size(); ++j) {
        replies[j] = recv_requests[j] + 1000;
    }

    // Reverse: feed the reversed layout into a second alltoallv (no re-negotiation).
    dstl::exchange_layout const rev = dstl::reverse_layout(layout);
    std::vector<int>            back(static_cast<std::size_t>(rev.recv_total()));
    kamping::v2::alltoallv(
        replies | views::with_counts(rev.send_counts) | views::with_displs(rev.send_displs),
        back | views::with_counts(rev.recv_counts) | views::with_displs(rev.recv_displs)
    );

    ASSERT_EQ(back.size(), data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(back[i], data[i] + 1000);
    }
}

// ── request_reply: unordered (plain recv_buffer, grouped by responder) ───────────────────────────────

TEST(RequestReply, UnorderedGrouped) {
    int  rank  = world_rank();
    int  size  = world_size();
    auto reqs  = build_pairs(rank, size);
    auto reply = [](int v) { return v + 1; };

    std::vector<int> result;
    dstl::request_reply(reqs, result | views::resize, reply);

    auto expected = expected_grouped<int>(reqs, size, reply);
    EXPECT_EQ(result, expected);
    EXPECT_EQ(sorted(result), sorted(expected)); // multiset also holds
}

// A pre-sized output buffer (plain recv_buffer, no resize view).
TEST(RequestReply, PreSizedOutput) {
    int  rank  = world_rank();
    int  size  = world_size();
    auto reqs  = build_pairs(rank, size);
    auto reply = [](int v) { return v * 5; };

    std::vector<int> result(reqs.size());
    dstl::request_reply(reqs, result, reply);

    EXPECT_EQ(result, (expected_grouped<int>(reqs, size, reply)));
}

// The returned (rvalue, owned) output carries the data out.
TEST(RequestReply, ReturnedOwnedOutput) {
    int  rank  = world_rank();
    int  size  = world_size();
    auto reqs  = build_pairs(rank, size);
    auto reply = [](int v) { return v - 7; };

    auto             res = dstl::request_reply(reqs, std::vector<int>{} | views::resize, reply);
    std::vector<int> got(std::ranges::begin(res), std::ranges::end(res));

    EXPECT_EQ(got, (expected_grouped<int>(reqs, size, reply)));
}

// ── request_reply: ordered_by_source (variadic recv_buffer_v) ────────────────────────────────────────

TEST(RequestReply, OrderedVariadic) {
    int  rank  = world_rank();
    int  size  = world_size();
    auto reqs  = build_pairs(rank, size);
    auto reply = [](int v) { return v + 1; };

    std::vector<int> result;
    dstl::request_reply(reqs, result | views::auto_recv_v, reply, MPI_COMM_WORLD, dstl::execution_policy::seq{},
                        dstl::layout::ordered_by_source{});

    EXPECT_EQ(result, (expected_grouped<int>(reqs, size, reply)));
}

// ── par policy: element-identical to seq ────────────────────────────────────────────────────────────

TEST(RequestReply, ParMatchesSeq) {
    int  rank  = world_rank();
    int  size  = world_size();
    auto reqs  = build_pairs(rank, size);
    auto reply = [](int v) { return v * 3 - 5; };

    std::vector<int> seq;
    std::vector<int> par;
    dstl::request_reply(reqs, seq | views::resize, reply, MPI_COMM_WORLD, dstl::execution_policy::seq{});
    dstl::request_reply(reqs, par | views::resize, reply, MPI_COMM_WORLD, dstl::execution_policy::par{});

    EXPECT_EQ(seq, par);
    EXPECT_EQ(seq, (expected_grouped<int>(reqs, size, reply)));
}

// ── datatype: a trivially-copyable struct reply whose MPI type rides on the recv buffer ─────────────

struct Pair {
    int  a;
    int  b;
    bool operator==(Pair const&) const = default;
};

// The request value is a builtin int (its type rides the send buffer); the reply is a struct whose MPI
// datatype is supplied on the output recv buffer via views::with_type.
TEST(RequestReply, StructReplyTypeFromBuffer) {
    int  rank  = world_rank();
    int  size  = world_size();
    auto reqs  = build_pairs(rank, size);
    auto reply = [](int v) { return Pair{v, v + 1}; };

    MPI_Datatype pair_dt = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(static_cast<int>(sizeof(Pair)), MPI_BYTE, &pair_dt);
    MPI_Type_commit(&pair_dt);

    std::vector<Pair> result;
    dstl::request_reply(reqs, result | views::resize | views::with_type(pair_dt), reply);

    auto const expected = expected_grouped<Pair>(reqs, size, reply);
    EXPECT_EQ(result, expected);

    MPI_Type_free(&pair_dt);
}

// ── edge cases ──────────────────────────────────────────────────────────────────────────────────────

TEST(RequestReply, AllEmpty) {
    std::vector<std::pair<int, int>> reqs;
    auto                             reply = [](int v) { return v + 1; };

    std::vector<int> result;
    dstl::request_reply(reqs, result | views::resize, reply);
    EXPECT_TRUE(result.empty());
}

TEST(RequestReply, UniformSingleElement) {
    int  rank  = world_rank();
    int  size  = world_size();
    auto reply = [](int v) { return v + 100; };

    std::vector<std::pair<int, int>> reqs;
    for (int j = 0; j < size; ++j) {
        reqs.emplace_back(rank * 10 + j, j); // one request to each rank
    }

    std::vector<int> result;
    dstl::request_reply(reqs, result | views::resize, reply);
    EXPECT_EQ(result, (expected_grouped<int>(reqs, size, reply)));
}
