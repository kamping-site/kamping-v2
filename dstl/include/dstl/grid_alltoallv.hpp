// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <mpi.h>

#include "dstl/default_init_allocator.hpp"
#include "dstl/detail/alltoallv_common.hpp"
#include "dstl/grid_comm.hpp"
#include "dstl/tags.hpp"
#include "kamping/kassert/kassert.hpp"
#include "kamping/v2/result.hpp"
#include "kamping/v2/views.hpp"
#include "mpi/buffer.hpp"
#include "mpi/collectives/alltoall.hpp"
#include "mpi/collectives/alltoallv.hpp"

/// @file
/// dstl::alltoallv — a k-dimensional grid (message-combining) all-to-all-v.
///
/// The routing replaces the p−1 direct messages per PE of a flat MPI_Alltoallv with k phases of
/// Σs_i messages, routed through the grid's subcommunicators. The flat per-destination send counts
/// fully determine the routing: they form a radix count-tree with p leaves, and each phase exchanges
/// one level of that tree (count metadata is interleaved with the data).
///
/// Recv ordering: `dstl::unordered` (default) delivers the correct multiset grouped by routing
/// path — this is where the speedup lives. `dstl::ordered_by_source` carries a per-element source
/// label through every hop and performs a final local stable sort so the result is byte-identical to
/// a flat MPI_Alltoallv.

namespace dstl {

/// Send buffer accepted by the grid alltoallv. On top of the standard variadic send-buffer contract
/// (data + type + per-destination counts + displacements) we additionally require that `ptr()`
/// exposes a concrete, copyable element type — the routing relocates whole element blocks locally
/// with `std::copy_n`, so a `void`-erased element type would make that copying ill-defined.
template <typename SBuf>
concept grid_send_buffer = mpi::experimental::send_buffer_v<std::remove_cvref_t<SBuf>>
                           && (!std::is_void_v<detail::element_t<SBuf>>) && std::copyable<detail::element_t<SBuf>>;

/// Recv buffer accepted by the grid alltoallv. Unlike the flat wrapper, the grid does *not* need a
/// deferred variadic buffer: it computes the total locally and writes one contiguous block. Hence the
/// recv buffer is simply a resizable, contiguous range whose element type matches the send buffer's.
template <typename RBuf>
concept grid_recv_buffer = std::ranges::contiguous_range<std::remove_cvref_t<RBuf>>
                           && std::copyable<std::ranges::range_value_t<std::remove_cvref_t<RBuf>>>
                           && requires(std::remove_cvref_t<RBuf>& r, std::size_t n) { r.resize(n); };

namespace detail {

/// State carried between phases of the grid routing.
/// `data` holds the elements ordered by *remaining destination index*; `dest_counts[t]` is the number
/// of elements whose remaining destination index is t (the current level of the radix count-tree).
/// When ordering by source, `source_rank[j]` is the global source rank of element j (routed in
/// lockstep with the data).
// `data` and `source_rank` are fully overwritten on every phase (MPI recv, then a copy_n rebin), so
// they use `uninit_vector` to skip the redundant value-initialization (see default_init_allocator.hpp).
// `dest_counts` stays a plain vector: it is accumulated into (zero-init then +=), so it must be value-
// initialized.
template <typename T>
struct routing_state {
    uninit_vector<T>   data;
    std::vector<int>   dest_counts; // size == product of not-yet-routed dimensions (current tree level)
    uninit_vector<int> source_rank; // size == data.size() (only when ordering by source)
};
// See grid_comm.hpp for a k = 2 diagram of how ranks map onto the grid and its subcommunicators, and
// the remote-first routing order this algorithm follows over them.

/// Run one phase (dimension `i`) of the grid routing: exchange the next count-tree level and the
/// corresponding data within subcommunicator `i`, then rebin the received blocks back into
/// remaining-destination-index order. `ParallelRebin` selects the OpenMP-parallel merge kernel.
template <typename T, bool Ordered, bool ParallelRebin>
void route_phase(routing_state<T>& state, std::size_t dim_size, MPI_Datatype dt, MPI_Comm subcomm) {
    namespace views = kamping::v2::views;

    auto const        subcomm_size      = static_cast<int>(dim_size);
    std::size_t const subtree_size      = state.dest_counts.size(); // size of the current count-tree level
    std::size_t const next_subtree_size = subtree_size / dim_size;  // each rank's subtree (next level's size)

    // A phase regroups (state.data, state.dest_counts): in goes data keyed by THIS dimension's
    // destination rank, out comes data keyed by the NEXT count-tree level. With s = subcomm_size and
    // m = next_subtree_size, the current level holds subtree_size = s*m buckets, and a remaining
    // destination index splits as idx = subcomm_rank * m + bucket. Write c[b,r] / b[b,r] for the
    // count / data of bucket b within destination rank r's block.
    //
    // BEFORE — sorted by remaining dest index; dest_counts is the current tree level (one int per
    // bucket). The m buckets of each rank r form the contiguous block sent to it, with
    // send_counts[r] = sum_b c[b,r]:
    //
    //                     dest rank 0                dest rank 1              ...     dest rank (subcomm_size -1)
    //                   +---------------------------+---------------------------+-----+
    // state.dest_counts:| c[0,0] c[1,0] .. c[m-1,0] | c[0,1] c[1,1] .. c[m-1,1] | ... |   subtree_size many entries
    // state.data:       | b[0,0] b[1,0] .. b[m-1,0] | b[0,1] b[1,1] .. b[m-1,1] | ... |   b[b,r] holds c[b,r] elems
    //                   +---------------------------+---------------------------+-----+
    //
    //   Note: c[i,j] = state.dest_counts[j*m + i]
    //
    //
    // AFTER EXCHANGE — the alltoallv ships each rank's block to that rank, so recv_data keeps the
    // block shape but is now keyed by SOURCE rank. Source src delivers recv_counts[src] =
    // sum_b recv_meta[b,src] elements, still ordered by bucket inside; recv_meta (from the count
    // alltoall above) carries the matching per-bucket counts in the same layout. Write e[b,src] for
    // the elements of bucket b that arrived from source src:
    //
    //                 source 0                   source 1                 ...     source (subcomm_size -1)
    //               +---------------------------+---------------------------+-----+
    // recv_data:    | e[0,0] e[1,0] .. e[m-1,0] | e[0,1] e[1,1] .. e[m-1,1] | ... |   e[b,src] holds recv_meta[b,src]
    //               +---------------------------+---------------------------+-----+
    //
    // AFTER MERGE — the merge transposes source<->bucket, concatenating each bucket across all
    // sources, so state.data is now sorted by the next dest index (the bucket) and dest_counts
    // collapses to the next tree level of m buckets, with next_dest_counts[b] = sum_src recv_meta[b,src]:
    //
    //                     bucket 0                   bucket 1                 ...     bucket (m-1)
    //                   +---------------------------+---------------------------+-----+
    // state.dest_counts:| next_dest_counts[0]       | next_dest_counts[1]       | ... |   = sum_src recv_meta[b,src]
    // state.data:       | e[0,0] e[0,1] .. e[0,s-1] | e[1,0] e[1,1] .. e[1,s-1] | ... |   bucket b gathered over srcs
    //                   +---------------------------+---------------------------+-----+
    //
    // These m buckets are the next count-tree level, already laid out for the next phase. With
    // m' = m / s' (s' = the next dimension's subcomm size), the buckets split into s' contiguous
    // ranges, one per destination of the next phase, so this AFTER picture is already that phase's
    // BEFORE picture:
    //
    // next dest:   next dest rank 0        next dest rank 1         ...  next dest rank s'-1
    //              [bucket 0, bucket m')   [bucket m', bucket 2m')  ...  [bucket (s'-1)m', bucket m)
    //
    std::vector<int> send_counts(dim_size, 0);
    for (std::size_t subcomm_rank = 0; subcomm_rank < dim_size; ++subcomm_rank) {
        auto const next_subtree_range_start = subcomm_rank * next_subtree_size;
        auto const subtree        = std::span{state.dest_counts}.subspan(next_subtree_range_start, next_subtree_size);
        send_counts[subcomm_rank] = std::accumulate(subtree.begin(), subtree.end(), 0);
    }
    std::vector<int> send_displs = exclusive_scan_int(send_counts);

    // Count metadata: ship the whole tree level, next_subtree_size ints per rank.
    // The receiver learns, per source rank and bucket, how many elements arrive.
    std::vector<int> recv_meta(subtree_size, 0);
    mpi::experimental::alltoall(state.dest_counts, recv_meta, subcomm);

    std::vector<int> recv_counts(static_cast<std::size_t>(subcomm_size), 0);
    for (int subcomm_rank = 0; subcomm_rank < subcomm_size; ++subcomm_rank) {
        int sum = 0;
        for (std::size_t bucket = 0; bucket < next_subtree_size; ++bucket) {
            sum += recv_meta[static_cast<std::size_t>(subcomm_rank) * next_subtree_size + bucket];
        }
        recv_counts[static_cast<std::size_t>(subcomm_rank)] = sum;
    }
    std::vector<int> recv_displs = exclusive_scan_int(recv_counts);
    int const        total_recv  = subcomm_size > 0 ? recv_displs[static_cast<std::size_t>(subcomm_size) - 1]
                                                          + recv_counts[static_cast<std::size_t>(subcomm_size) - 1]
                                                    : 0;

    // recv_counts / recv_displs were derived locally from recv_meta above (the BEFORE half of the
    // phase diagram at the top of this function), so we attach them explicitly rather than let the
    // kamping::v2 layer re-negotiate them (a redundant collective per phase).
    // with_type(dt) preserves the caller's MPI datatype; when ordering by source, source_rank rides
    // along with the identical counts and displacements.
    uninit_vector<T> recv_data(static_cast<std::size_t>(total_recv));
    mpi::experimental::alltoallv(
        state.data | views::with_type(dt) | views::with_counts(send_counts) | views::with_displs(send_displs),
        recv_data | views::with_type(dt) | views::with_counts(recv_counts) | views::with_displs(recv_displs),
        subcomm
    );
    uninit_vector<int> recv_source_rank;
    if constexpr (Ordered) {
        recv_source_rank.resize(static_cast<std::size_t>(total_recv));
        mpi::experimental::alltoallv(
            state.source_rank | views::with_counts(send_counts) | views::with_displs(send_displs),
            recv_source_rank | views::with_counts(recv_counts) | views::with_displs(recv_displs),
            subcomm
        );
    }

    // Rebin: merge the per-source-rank blocks back into remaining-index order. This is the
    // execution-policy hotspot; each bucket is written independently.
    std::vector<int> next_dest_counts(next_subtree_size, 0);
    for (std::size_t subcomm_rank = 0; subcomm_rank < static_cast<std::size_t>(subcomm_size); ++subcomm_rank) {
        for (std::size_t bucket = 0; bucket < next_subtree_size; ++bucket) {
            next_dest_counts[bucket] += recv_meta[subcomm_rank * next_subtree_size + bucket];
        }
    }
    std::vector<int> new_displs = exclusive_scan_int(next_dest_counts);

    // Read offset (in elements) of the (source rank, bucket) sub-block within the received block.
    std::vector<int> read_off(subtree_size, 0);
    for (int subcomm_rank = 0; subcomm_rank < subcomm_size; ++subcomm_rank) {
        int cursor = recv_displs[static_cast<std::size_t>(subcomm_rank)];
        for (std::size_t bucket = 0; bucket < next_subtree_size; ++bucket) {
            auto const slot = static_cast<std::size_t>(subcomm_rank) * next_subtree_size + bucket;
            read_off[slot]  = cursor;
            cursor += recv_meta[slot];
        }
    }

    auto merge = [&]<typename Src>(Src const& src) {
        using U = std::ranges::range_value_t<Src>;
        uninit_vector<U> dst(static_cast<std::size_t>(total_recv));
        // Each bucket writes its own disjoint output slice (new_displs[bucket]…), so the buckets split
        // safely into one contiguous chunk per thread.
        chunked_for(ParallelRebin, static_cast<std::ptrdiff_t>(next_subtree_size), [&](std::ptrdiff_t lo, std::ptrdiff_t hi) {
            for (std::ptrdiff_t bucket = lo; bucket < hi; ++bucket) {
                auto write = static_cast<std::size_t>(new_displs[static_cast<std::size_t>(bucket)]);
                for (int subcomm_rank = 0; subcomm_rank < subcomm_size; ++subcomm_rank) {
                    auto const slot =
                        static_cast<std::size_t>(subcomm_rank) * next_subtree_size + static_cast<std::size_t>(bucket);
                    int const cnt = recv_meta[slot];
                    if (cnt > 0) {
                        auto const from = static_cast<std::size_t>(read_off[slot]);
                        std::copy_n(
                            src.begin() + static_cast<std::ptrdiff_t>(from),
                            cnt,
                            dst.begin() + static_cast<std::ptrdiff_t>(write)
                        );
                        write += static_cast<std::size_t>(cnt);
                    }
                }
            }
        });
        return dst;
    };

    state.data = merge(recv_data);
    if constexpr (Ordered) {
        state.source_rank = merge(recv_source_rank);
    }
    state.dest_counts = std::move(next_dest_counts);
}

/// Core implementation shared by the free function and the member entry points.
template <execution_policy Exec, typename Order, grid_send_buffer SBuf, grid_recv_buffer RBuf>
void grid_alltoallv_impl(SBuf const& sbuf, RBuf& rbuf, grid_comm<Exec> const& grid, Order /* order */) {
    using T = element_t<SBuf>;
    static_assert(
        std::is_same_v<T, std::ranges::range_value_t<RBuf>>,
        "dstl::alltoallv: send and recv buffers must have the same element type."
    );
    constexpr bool ordered  = std::is_same_v<Order, ordered_by_source>;
    constexpr bool parallel = !std::is_same_v<Exec, sequential>;

    auto const p  = static_cast<std::size_t>(grid.size());
    auto const dt = mpi::experimental::type(sbuf);

    auto const  send_counts = mpi::experimental::counts(sbuf); // span<int const>, size p
    auto const  send_displs = mpi::experimental::displs(sbuf); // span<int const>, size p
    auto const* src         = mpi::experimental::ptr(sbuf);    // T const*

    // Initial state: compact the send buffer into the count-tree's leaf order. With the row-major
    //    grid_comm convention the global-rank order IS the leaf order — dimension 0 (the largest
    //    stride, most "remote" dimension) is the most-significant digit — so the radix count-tree
    //    falls out directly: the flat per-destination counts ARE the root tree, and phase 0 peels
    //    dimension 0 because its members already form contiguous blocks of the globally-ordered
    //    send buffer. No permutation is needed.
    //
    // TODO: route LOCAL-first instead of remote-first. On a hierarchical machine, combining within
    //    the most local (smallest-stride, intra-node) subcommunicator before the expensive remote
    //    hops lets the later tiers carry fewer, larger messages. Switching directions needs only two
    //    changes — the per-phase machinery (route_phase) is unchanged:
    //      1. Re-lay this initial buffer into reversed-digit leaf order, making the local dimension
    //         (k-1, stride 1) the most-significant leaf digit:
    //             leaf(d) = Σ_i coords(d)[i] · leaf_stride[i],  leaf_stride[i] = Π_{j>i} dims[j],
    //         and build tree/data by that permutation. (The send buffer arrives in global-rank order,
    //         whose most-significant digit is the most-remote dimension, so this reorder is the price
    //         of going the other way.)
    //      2. Iterate the phase loop and subcommunicators in reverse (i = k-1 … 0).
    routing_state<T> state;
    state.dest_counts.assign(send_counts.begin(), send_counts.end());
    int total_send = 0;
    for (std::size_t d = 0; d < p; ++d) {
        total_send += send_counts[d];
    }
    state.data.resize(static_cast<std::size_t>(total_send));
    {
        int write = 0;
        for (std::size_t d = 0; d < p; ++d) {
            int const cnt = send_counts[d];
            if (cnt > 0) {
                std::copy_n(src + static_cast<std::ptrdiff_t>(send_displs[d]), cnt, state.data.begin() + write);
                write += cnt;
            }
        }
    }
    if constexpr (ordered) {
        state.source_rank.assign(static_cast<std::size_t>(total_send), grid.rank());
    }

    // k phases, remote dimension first (dimension 0 → k-1).
    for (std::size_t i = 0; i < grid.num_dims(); ++i) {
        route_phase<T, ordered, parallel>(state, grid.dim_size(i), dt, grid.subcomm(i).mpi_handle());
    }

    int const total_recv = state.dest_counts.empty() ? 0 : state.dest_counts[0];

    // Deposit into the recv buffer (resize + one contiguous write).
    rbuf.resize(static_cast<std::size_t>(total_recv));
    auto* out = std::ranges::data(rbuf);
    if constexpr (ordered) {
        // Group by global source rank, preserving within-source order (stable) so the layout is
        // byte-identical to a flat MPI_Alltoallv.
        std::vector<int> order_idx(static_cast<std::size_t>(total_recv));
        std::iota(order_idx.begin(), order_idx.end(), 0);
        std::ranges::stable_sort(order_idx, [&](int a, int b) {
            return state.source_rank[static_cast<std::size_t>(a)] < state.source_rank[static_cast<std::size_t>(b)];
        });
        for (std::size_t pos = 0; pos < order_idx.size(); ++pos) {
            out[pos] = state.data[static_cast<std::size_t>(order_idx[pos])];
        }
    } else {
        // Flat contiguous deposit: each chunk memcpys its own disjoint slice via std::copy_n.
        chunked_for(parallel, static_cast<std::ptrdiff_t>(total_recv), [&](std::ptrdiff_t lo, std::ptrdiff_t hi) {
            std::copy_n(state.data.begin() + lo, hi - lo, out + lo);
        });
    }
}

} // namespace detail

/// Drop-in grid all-to-all-v (D1). The call site matches the flat KaMPIng v2 alltoallv; only the
/// communicator changes from a flat comm to a `grid_comm`. The send buffer is a standard variadic
/// `send_buffer_v` (`data | with_counts | with_displs`); the recv buffer is any resizable contiguous
/// range (e.g. a plain `std::vector<T>`).
///
/// @tparam Order Recv ordering tag: `unordered` (default) or `ordered_by_source`.
template <grid_send_buffer SBuf, grid_recv_buffer RBuf, execution_policy Exec, recv_ordering Order = unordered>
auto alltoallv(SBuf&& sbuf, RBuf&& rbuf, grid_comm<Exec> const& grid, Order order = {})
    -> kamping::v2::result<SBuf, RBuf> {
    kamping::v2::result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    detail::grid_alltoallv_impl(res.send, res.recv, grid, order);
    return res;
}
} // namespace dstl
