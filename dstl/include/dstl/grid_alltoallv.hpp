// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

#ifdef __linux__
    #include <malloc.h>
#endif

#include <mpi.h>

#include "dstl/default_init_allocator.hpp"
#include "dstl/detail/alltoallv_common.hpp"
#include "dstl/detail/recv_buffer_utils.hpp"
#include "dstl/grid_comm.hpp"
#include "dstl/tags.hpp"
#include "kamping/kassert/kassert.hpp"
#include "kamping/v2/collectives/alltoall.hpp"
#include "kamping/v2/collectives/alltoallv.hpp"
#include "kamping/v2/result.hpp"
#include "kamping/v2/views.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "mpi/buffer.hpp"
#include "mpi/comm.hpp"

/// @file
/// dstl::grid_alltoallv — a k-dimensional grid (message-combining) all-to-all-v.
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

namespace detail {
/// Element-type requirements shared by the grid send/recv buffers. Beyond an ordinary alltoallv the
/// routing only relocates whole element blocks locally — it `std::copy_n`s them through internal
/// staging buffers — so the element type need satisfy exactly what that staging needs:
///   * `std::indirectly_copyable<T const*, T*>` — the precondition of `std::copy_n`
///     (https://en.cppreference.com/w/cpp/algorithm/ranges/copy). `copy_n` *assigns* into existing
///     elements, so this is strictly weaker than `std::copyable` (no copy-construction required), and
///     a `void`-erased `ptr()` fails it for free (`void const*` is not `indirectly_readable`).
///   * `std::default_initializable<T>` — the staging `uninit_vector<T>` default-initializes its
///     elements before they are overwritten.
template <typename T>
concept grid_element = std::default_initializable<T> && std::indirectly_copyable<T const*, T*>;
} // namespace detail

/// Send buffer accepted by the grid alltoallv: a standard variadic send buffer (`send_buffer_v` — data +
/// type + per-destination counts + displacements) that is additionally a `contiguous_range`. The range
/// requirement is what makes the element type extractable: the buffer protocol only guarantees `ptr()` is
/// convertible to `void const*`, so the element type cannot be read off the pointer — we recover it via
/// `std::ranges::range_value_t`. Contiguity (not just `range`) is the honest contract: the body does
/// pointer arithmetic on `ptr()` and stages whole elements, and a contiguous `T` array whose stride equals
/// the datatype's extent is exactly what makes a *gapped* MPI datatype behave correctly. The element type
/// must satisfy `detail::grid_element` (default-initializable + copyable via `std::ranges::copy_n` for the
/// internal staging). (A `send_buffer_v` that is not a contiguous range — e.g. a raw `mpi_span_v` — has no
/// knowable element type and so cannot drive the local block copies the routing performs; it is correctly
/// rejected.)
template <typename SBuf>
concept grid_send_buffer = mpi::experimental::send_buffer_v<SBuf> && std::ranges::contiguous_range<SBuf>
                           && detail::grid_element<std::ranges::range_value_t<SBuf>>;

/// Concept: a recv-layout tag the grid alltoallv knows how to produce — either `layout::unordered`
/// (multiset-correct, grouped by routing path) or `layout::ordered_by_source` (flat-identical).
template <typename T>
concept grid_compatible_output_layout =
    std::is_same_v<T, layout::unordered> || std::is_same_v<T, layout::ordered_by_source>;

/// Recv buffer accepted by the grid alltoallv, constrained by the recv ordering. In `ordered_by_source`
/// mode the result is a genuine alltoallv with well-defined per-source counts, so the recv buffer must be
/// a *variadic* recv buffer (`recv_buffer_v`) able to carry them (e.g. `views::auto_recv_v`); in
/// `unordered` mode only the multiset is defined, so an ordinary `recv_buffer` (e.g. a plain
/// `std::vector<T>` or `views::resize`) suffices. In both cases the buffer must be a `contiguous_range`, so
/// its element type is extractable via `std::ranges::range_value_t` (see `grid_send_buffer`) and must
/// satisfy `detail::grid_element`. The send and recv element types need NOT be identical:
///   * `unordered` fills `rbuf` directly via MPI, so different send/recv datatypes are allowed exactly like
///     a plain `MPI_Alltoallv` (matched by type *signature*, not extent) — no host-side relation required;
///   * `ordered_by_source` regroups with a local counting-sort copy, so the function additionally requires
///     `std::indirectly_copyable<iterator_t<SBuf>, iterator_t<RBuf>>`.
/// Sizing — and, for `ordered_by_source`, the per-source counts — are handled through the kamping v2
/// machinery in the body (see `ensure_recv_capacity` / `size_from_source_counts`).
template <typename RBuf, typename Order>
concept grid_recv_buffer = std::ranges::contiguous_range<RBuf>
                           && ((std::is_same_v<Order, layout::ordered_by_source> && mpi::experimental::recv_buffer_v<RBuf>)
                               || (!std::is_same_v<Order, layout::ordered_by_source> && mpi::experimental::recv_buffer<RBuf>))
                           && detail::grid_element<std::ranges::range_value_t<RBuf>>;

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
//
// The recv-buffer sizing helpers `ensure_recv_capacity` and `size_from_source_counts` live in
// dstl/detail/recv_buffer_utils.hpp (shared with the reversible exchange / request_reply path).

/// Group the routed elements by their global source rank into `rbuf`, byte-identical to a flat
/// MPI_Alltoallv. This is a stable counting sort (O(n), not O(n log n)): histogram the source labels,
/// turn the per-source totals into base write offsets (`size_from_source_counts`), then scatter each
/// element into its source's block — elements of the same source keep their arrival order.
///
/// `Parallel` selects an OpenMP two-pass kernel over a fixed `chunking` of the input. The chunk *index*
/// — not the thread id — keys the per-chunk histograms and write cursors, and `chunking` recomputes the
/// chunk bounds identically in both passes, so correctness does not depend on the OpenMP schedule (a
/// count/write thread mismatch under e.g. `dynamic` is harmless). Within a source, chunk `c` writes after
/// every earlier chunk's elements of that source, so chunks stay ordered and the sort is stable. To avoid
/// false sharing, each chunk's hot count/scatter loop touches only a thread-private vector (`local`
/// histogram / `cursor` copy), never memory shared with a sibling.
template <typename T, bool Parallel, typename RBuf>
void reorder_by_source_rank(
    RBuf& rbuf, std::span<T const> recv_data, std::span<int const> recv_source_rank, int global_size
) {
    auto const     total_recv = static_cast<int>(recv_data.size());
    chunking const chunks{Parallel, static_cast<std::ptrdiff_t>(total_recv)};

    // Pass 1 — per-chunk source-label histograms. Each chunk counts into its own thread-private vector
    // and only stores it (by move) afterwards, so the hot loop never shares a cache line with a sibling.
    std::vector<std::vector<int>> chunk_counts(static_cast<std::size_t>(chunks.count()));
    chunks.for_each_chunk([&](int c, std::ptrdiff_t lo, std::ptrdiff_t hi) {
        std::vector<int> local(static_cast<std::size_t>(global_size), 0);
        for (auto j = lo; j < hi; ++j) {
            ++local[static_cast<std::size_t>(recv_source_rank[static_cast<std::size_t>(j)])];
        }
        chunk_counts[static_cast<std::size_t>(c)] = std::move(local);
    });

    // Global per-source totals (= the variadic recv buffer's recv_counts).
    std::vector<int> totals(static_cast<std::size_t>(global_size), 0);
    for (int c = 0; c < chunks.count(); ++c) {
        for (int s = 0; s < global_size; ++s) {
            totals[static_cast<std::size_t>(s)] +=
                chunk_counts[static_cast<std::size_t>(c)][static_cast<std::size_t>(s)];
        }
    }

    // Size rbuf and obtain each source's base write offset (its displacement).
    std::vector<int> const base = size_from_source_counts(rbuf, std::span<int const>{totals}, total_recv);

    // Per-chunk write cursors: within each source, chunk c starts after all earlier chunks' elements of
    // that source (a prefix sum over chunks), keeping chunks ordered for a stable result.
    std::vector<std::vector<int>> write_pos(
        static_cast<std::size_t>(chunks.count()),
        std::vector<int>(static_cast<std::size_t>(global_size))
    );
    for (int s = 0; s < global_size; ++s) {
        int running = base[static_cast<std::size_t>(s)];
        for (int c = 0; c < chunks.count(); ++c) {
            write_pos[static_cast<std::size_t>(c)][static_cast<std::size_t>(s)] = running;
            running += chunk_counts[static_cast<std::size_t>(c)][static_cast<std::size_t>(s)];
        }
    }

    // Pass 2 — scatter. Each chunk copies its cursors into a thread-private vector first, so the
    // per-element cursor increments never share a cache line with another thread.
    auto* out = mpi::experimental::ptr(rbuf);
    chunks.for_each_chunk([&](int c, std::ptrdiff_t lo, std::ptrdiff_t hi) {
        std::vector<int> cursor = write_pos[static_cast<std::size_t>(c)];
        for (auto j = lo; j < hi; ++j) {
            auto const s = static_cast<std::size_t>(recv_source_rank[static_cast<std::size_t>(j)]);
            out[static_cast<std::size_t>(cursor[s]++)] = recv_data[static_cast<std::size_t>(j)];
        }
    });
}

/// Rebin a phase's received blocks back into remaining-destination-index order: the alltoallv delivers
/// data keyed by source rank (each source's block holding its `next_subtree_size` buckets in order), and
/// this transposes source<->bucket so the result is keyed by the next count-tree level (the bucket),
/// concatenating each bucket across all sources. Writes the rebinned data (and, when `Ordered`, the
/// source labels) and the next tree level back into `state`. `recv_meta[src * next_subtree_size + bucket]`
/// is the element count of that (source, bucket) sub-block and `recv_displs[src]` the base offset of
/// source `src`'s received block. This is the execution-policy hotspot: each bucket writes its own
/// disjoint output slice (`new_displs[bucket]…`), so `ParallelRebin` splits the buckets into one
/// contiguous chunk per OpenMP thread.
template <typename T, bool Ordered, bool ParallelRebin>
void rebin(
    routing_state<T>&         state,
    uninit_vector<T> const&   recv_data,
    uninit_vector<int> const& recv_source_rank,
    std::span<int const>      recv_meta,
    std::span<int const>      recv_displs,
    int                       subcomm_size,
    std::size_t               next_subtree_size,
    int                       total_recv
) {
    std::size_t const subtree_size = static_cast<std::size_t>(subcomm_size) * next_subtree_size;

    std::vector<int> next_dest_counts(next_subtree_size, 0);
    for (std::size_t subcomm_rank = 0; subcomm_rank < static_cast<std::size_t>(subcomm_size); ++subcomm_rank) {
        for (std::size_t bucket = 0; bucket < next_subtree_size; ++bucket) {
            next_dest_counts[bucket] += recv_meta[subcomm_rank * next_subtree_size + bucket];
        }
    }
    std::vector<int> new_displs(next_dest_counts.size());
    std::exclusive_scan(next_dest_counts.begin(), next_dest_counts.end(), new_displs.begin(), 0);

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
        chunked_for(
            ParallelRebin,
            static_cast<std::ptrdiff_t>(next_subtree_size),
            [&](std::ptrdiff_t lo, std::ptrdiff_t hi) {
                for (std::ptrdiff_t bucket = lo; bucket < hi; ++bucket) {
                    auto write = static_cast<std::size_t>(new_displs[static_cast<std::size_t>(bucket)]);
                    for (int subcomm_rank = 0; subcomm_rank < subcomm_size; ++subcomm_rank) {
                        auto const slot = static_cast<std::size_t>(subcomm_rank) * next_subtree_size
                                          + static_cast<std::size_t>(bucket);
                        int const  cnt  = recv_meta[slot];
                        if (cnt > 0) {
                            auto const from = static_cast<std::size_t>(read_off[slot]);
                            std::ranges::copy_n(
                                src.begin() + static_cast<std::ptrdiff_t>(from),
                                cnt,
                                dst.begin() + static_cast<std::ptrdiff_t>(write)
                            );
                            write += static_cast<std::size_t>(cnt);
                        }
                    }
                }
            }
        );
        return dst;
    };

    state.data = merge(recv_data);
    if constexpr (Ordered) {
        state.source_rank = merge(recv_source_rank);
    }
    state.dest_counts = std::move(next_dest_counts);
}

/// Run one phase (dimension `i`) of the grid routing: exchange the next count-tree level and the
/// corresponding data within subcommunicator `i`, then rebin the received blocks back into
/// remaining-destination-index order. `ParallelRebin` selects the OpenMP-parallel merge kernel.
///
/// On the last phase (`is_last`) the count-tree has collapsed to a single bucket per rank
/// (`next_subtree_size == 1`), which makes the rebin an identity. We therefore skip the staging buffer
/// and the rebin and deposit the routed result straight into the caller's `rbuf` (sized here), fusing
/// the final receive with the result write-out (see D8). `global_size` is the grid's global comm size,
/// needed only to size a variadic deferred recv buffer in `ordered_by_source` mode.
template <typename T, bool Ordered, bool ParallelRebin, typename RBuf>
void route_phase(
    routing_state<T>&            state,
    std::size_t                  dim_size,
    MPI_Datatype                 dt,
    mpi::experimental::comm_view subcomm,
    RBuf&                        rbuf,
    bool                         is_last,
    [[maybe_unused]] int         global_size,
    long                         call_id
) {
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
    std::vector<int> send_displs(send_counts.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), 0);

    // Count metadata: ship the whole tree level, next_subtree_size ints per rank.
    // The receiver learns, per source rank and bucket, how many elements arrive.
    std::vector<int> recv_meta(subtree_size, 0);
    kamping::v2::alltoall(state.dest_counts, recv_meta, subcomm);

    std::vector<int> recv_counts(static_cast<std::size_t>(subcomm_size), 0);
    for (int subcomm_rank = 0; subcomm_rank < subcomm_size; ++subcomm_rank) {
        int sum = 0;
        for (std::size_t bucket = 0; bucket < next_subtree_size; ++bucket) {
            sum += recv_meta[static_cast<std::size_t>(subcomm_rank) * next_subtree_size + bucket];
        }
        recv_counts[static_cast<std::size_t>(subcomm_rank)] = sum;
    }
    std::vector<int> recv_displs(recv_counts.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);
    int const total_recv = subcomm_size > 0 ? recv_displs[static_cast<std::size_t>(subcomm_size) - 1]
                                                  + recv_counts[static_cast<std::size_t>(subcomm_size) - 1]
                                            : 0;

    // TEMPORARY DIAGNOSTIC (2026-09-01, per KaCCv2-side request -- there are only a
    // handful of phases per grid_alltoallv call (num_dims, single digits), so print the
    // full per-phase metadata unconditionally rather than gating everything behind an
    // assertion that might silently pass. subcomm.rank()/size() are phase-local (this
    // dimension's subcommunicator), not the global rank -- dim_size/is_last pin down
    // which phase this is, and KaCCv2's own bfs.hpp prints "a2a round=" immediately
    // around each grid_alltoallv call, so the two logs correlate by proximity even
    // without a shared round number. Revert once answered.
    //
    // Built into one string and written with a single fputs/fwrite: two ranks calling
    // fprintf multiple times each for the "same" line (as an earlier version of this
    // diagnostic did) interleave mid-line on a shared stderr stream under real
    // multi-node MPI, since each individual fprintf call is its own unsynchronized
    // write -- confirmed garbled in exactly that way in the first SuperMUC run using
    // this diagnostic. A single call's write of a short line is atomic on the streams
    // MPI redirects to (POSIX pipes/regular files under PIPE_BUF), so one string, one
    // call.
    // TEMPORARY DIAGNOSTIC (2026-09-02, continued -- KaCCv2's investigation established
    // that the corrupting call is never a fixed absolute call_id (call #17 was three
    // runs' coincidence; a later rerun landed on #24 instead) but IS always the same
    // total_recv size class (~6400 elements, BFS round 3) -- see
    // notes/grid-alltoallv-supermuc-data-loss.md's "size-class trigger, not fixed call
    // count" correction. This decouples the two: `occurrence` is the count of prior
    // calls (this rank, this process lifetime) with this exact total_recv value,
    // independent of call_id, num_dims, or how many other grid_alltoallv calls (e.g.
    // from a coloring phase that does real work) happened in between. A future crash
    // log line can then read off "this was the Nth time this rank saw a call this
    // size" directly, without needing an external confound-breaking suite config
    // (c9 tried and failed at this -- see notes). Keyed by exact total_recv, not a
    // rounded bucket, since the fixed seed makes every occurrence of the real bug
    // byte-identical in size. Revert once answered.
    static std::unordered_map<int, long> size_class_occurrences;
    long const occurrence = ++size_class_occurrences[total_recv];

    {
        std::string line = "[grid_alltoallv route_phase] call=" + std::to_string(call_id)
                          + " subrank=" + std::to_string(subcomm.rank())
                          + "/" + std::to_string(subcomm.size()) + " dim_size=" + std::to_string(dim_size)
                          + " is_last=" + std::to_string(is_last ? 1 : 0)
                          + " total_recv=" + std::to_string(total_recv)
                          + " size_class_occurrence=" + std::to_string(occurrence) + " send_counts=[";
        for (std::size_t i = 0; i < send_counts.size(); ++i) {
            line += std::to_string(send_counts[i]);
            if (i + 1 < send_counts.size()) line += ",";
        }
        line += "] recv_counts=[";
        for (std::size_t i = 0; i < recv_counts.size(); ++i) {
            line += std::to_string(recv_counts[i]);
            if (i + 1 < recv_counts.size()) line += ",";
        }
        line += "] recv_meta=[";
        for (std::size_t i = 0; i < recv_meta.size(); ++i) {
            line += std::to_string(recv_meta[i]);
            if (i + 1 < recv_meta.size()) line += ",";
        }
        line += "]\n";
        std::fputs(line.c_str(), stderr);
        std::fflush(stderr);
    }

    // TEMPORARY DIAGNOSTIC (2026-09-01, KaCCv2 grid-fanout investigation -- see that repo's
    // notes/frozen-multistep-coverage-and-imbalance-findings.md). A caller-side global
    // sent/received element-count cross-check around the whole grid_alltoallv call passed
    // (global totals balance) on a run that still hit "received a real message whose vertex
    // decodes to 0" downstream -- so the bug isn't a global over/under-count, and must be
    // either local mis-accounting that cancels out globally, or content corruption that
    // preserves counts. This isolates the first possibility: recv_counts (derived above from
    // recv_meta, itself carried by the kamping::v2::alltoall two lines up) is exactly the
    // "how much does each peer THINK it's sending me" side of a normal alltoall exchange.
    // Independently re-negotiate the same metadata via a redundant, ordinary MPI_Alltoall of
    // send_counts and assert it agrees. Disagreement here would mean the count-metadata
    // exchange itself (not the data exchange or rebin that follows) is the fault. Revert once
    // answered.
    {
        std::vector<int> verify_recv_counts(static_cast<std::size_t>(subcomm_size), 0);
        MPI_Alltoall(
            send_counts.data(), 1, MPI_INT, verify_recv_counts.data(), 1, MPI_INT, subcomm.mpi_handle()
        );
        for (int r = 0; r < subcomm_size; ++r) {
            KAMPING_V2_ASSERT(
                verify_recv_counts[static_cast<std::size_t>(r)] == recv_counts[static_cast<std::size_t>(r)],
                "grid_alltoallv route_phase: rank "
                    << subcomm.rank() << " (dim_size=" << dim_size << " is_last=" << is_last
                    << "): recv_counts[" << r << "]=" << recv_counts[static_cast<std::size_t>(r)]
                    << " but independently-negotiated MPI_Alltoall says "
                    << verify_recv_counts[static_cast<std::size_t>(r)]
            );
        }
    }

    // recv_counts / recv_displs were derived locally from recv_meta above (the BEFORE half of the
    // phase diagram at the top of this function), so we attach them explicitly rather than let the
    // kamping::v2 layer re-negotiate them (a redundant collective per phase).
    // with_type(dt) preserves the caller's MPI datatype.
    auto send_data =
        state.data | views::with_type(dt) | views::with_counts(send_counts) | views::with_displs(send_displs);

    // TEMPORARY DIAGNOSTIC (2026-09-01, KaCCv2 grid-fanout investigation, revised after a
    // static-audit subagent flagged two weaknesses in the first version of this dump:
    // (a) it printed all sizeof(T)=16 bytes of KaCC's wire Message
    // (std::pair<uint64_t VId, int Payload>), but that type's actual MPI datatype is
    // MPI_Type_create_struct{uint64_t@0, int@8} resized to 16 -- bytes 12-15 are a
    // padding hole MPI never transfers, so a byte-for-byte "match" there is 4 bytes of
    // coincidence and a mismatch confined to that range would be a false positive.
    // KACC_DIAG_ELEM_BYTES below is hardcoded to 12 for exactly this reason -- it is
    // NOT generically correct for any T, only for this investigation's specific element
    // type, and must be reverted along with the rest of this diagnostic.
    // (b) it dumped only element 0, and paired SEND/RECV lines across ranks purely by
    // log order, with no shared call/round identifier -- call_id (this function's new
    // parameter, a monotonic per-process counter incremented once per top-level
    // grid_alltoallv() call) fixes the pairing; dumping up to the first 4 elements
    // instead of just 1 shows whether any corruption is confined to position 0 or
    // spread across the block.
    //
    // Rationale for the dump itself, unchanged from the previous version: rank 0's own
    // bfs.hpp-side trace (this same investigation, separate log) shows a real, non-zero
    // element queued as the first message to the peer at exactly this round; the peer's
    // crash shows an all-zero element at the equivalent receive position, with every
    // count cross-check (global, per-phase, this branch's independent recv_counts
    // renegotiation) agreeing exactly -- so the corruption is content, not size or
    // routing, and must happen between here (about to cross the wire) and the matching
    // point just after the alltoallv call below. Revert once answered.
    constexpr std::size_t KACC_DIAG_ELEM_BYTES = 12;
    for (int r = 0; r < subcomm_size; ++r) {
        if (r == subcomm.rank() || send_counts[static_cast<std::size_t>(r)] == 0) continue;
        auto const  count       = static_cast<std::size_t>(send_counts[static_cast<std::size_t>(r)]);
        auto const  dump_count  = std::min<std::size_t>(count, 4);
        auto const* base        = reinterpret_cast<unsigned char const*>(
            &state.data[static_cast<std::size_t>(send_displs[static_cast<std::size_t>(r)])]
        );
        std::string line = "[grid_alltoallv route_phase SEND] call=" + std::to_string(call_id)
                          + " subrank=" + std::to_string(subcomm.rank()) + "/" + std::to_string(subcomm.size())
                          + " dim_size=" + std::to_string(dim_size) + " is_last=" + std::to_string(is_last ? 1 : 0)
                          + " to_subrank=" + std::to_string(r) + " count=" + std::to_string(count)
                          + " elems=[";
        for (std::size_t e = 0; e < dump_count; ++e) {
            auto const* elem = base + e * sizeof(T);
            std::string hex;
            for (std::size_t b = 0; b < KACC_DIAG_ELEM_BYTES; ++b) {
                char buf[4];
                std::snprintf(buf, sizeof(buf), "%02x", elem[b]);
                hex += buf;
            }
            line += hex;
            if (e + 1 < dump_count) line += ",";
        }
        line += "]\n";
        std::fputs(line.c_str(), stderr);
    }

    // Last phase: next_subtree_size == 1, so the rebin/merge below would be an identity (it just
    // concatenates each source's already-contiguous recv block in source order — exactly the alltoallv
    // recv layout). We skip the staging buffer and the post-loop result copy and deposit the routed
    // result straight into the caller's recv buffer (see D8 in DSTL-Alltoallv-Design.md).
    if (is_last) {
        if constexpr (!Ordered) {
            // Unordered: the received multiset IS the result. Opt-in resize only — for a deferred recv
            // buffer (views::resize) ensure_recv_capacity records the count and the alltoallv's ptr()
            // access materializes it; a pre-sized buffer is used as-is. We deposit straight into rbuf, no
            // staging buffer and no result copy. Crucially we do NOT override rbuf's datatype with dt
            // (= dt_send): rbuf keeps its own type (dt_recv), so the final exchange is
            // alltoallv(staging, dt_send, ..., rbuf, dt_recv) — send/recv types matched by signature,
            // exactly like a plain MPI_Alltoallv (extents may differ, e.g. gapped send -> packed recv).
            ensure_recv_capacity(rbuf, total_recv);
            kamping::v2::alltoallv(
                send_data,
                rbuf | views::with_counts(recv_counts) | views::with_displs(recv_displs),
                subcomm
            );
        } else {
            // Ordered: receive data + per-element source labels into temporaries, then group by global
            // source into rbuf (sizing it) so the layout is byte-identical to a flat MPI_Alltoallv.
            uninit_vector<T> recv_data(static_cast<std::size_t>(total_recv));
            kamping::v2::alltoallv(
                send_data,
                recv_data | views::with_type(dt) | views::with_counts(recv_counts) | views::with_displs(recv_displs),
                subcomm
            );
            uninit_vector<int> recv_source_rank(static_cast<std::size_t>(total_recv));
            kamping::v2::alltoallv(
                state.source_rank | views::with_counts(send_counts) | views::with_displs(send_displs),
                recv_source_rank | views::with_counts(recv_counts) | views::with_displs(recv_displs),
                subcomm
            );
            reorder_by_source_rank<T, ParallelRebin>(
                rbuf,
                std::span<T const>{recv_data},
                std::span<int const>{recv_source_rank},
                global_size
            );
        }
        return;
    }

    // Intermediate phase: stage the received blocks, then rebin them back into remaining-index order.
    // When ordering by source, source_rank rides along with the identical counts and displacements.
    uninit_vector<T> recv_data(static_cast<std::size_t>(total_recv));

    // TEMPORARY DIAGNOSTIC (2026-09-01, KaCCv2 grid-fanout investigation, added after a
    // static-audit subagent pointed out recv_data is genuinely UNINITIALIZED
    // (uninit_vector, see default_init_allocator.hpp) -- "the receiver got all-zero
    // bytes" is exactly what a slot MPI never wrote at all would also look like, once
    // heap pages happen to come back zeroed (e.g. after malloc_trim, which KaCCv2's
    // benchmark runner calls every iteration). Poisoning with a non-zero, non-plausible
    // pattern before the real alltoallv call turns that ambiguity into a direct
    // yes/no: if the RECV dump below ever shows this exact poison pattern instead of
    // either the correct content OR all-zero, MPI genuinely never wrote that slot --
    // proof of a sizing/count bug despite every count cross-check agreeing so far. If it
    // shows all-zero (not the poison pattern), MPI DID write something, and the
    // all-zero content is either genuine wire corruption or genuinely-zero real data --
    // still meaningful either way. Revert once answered.
    std::memset(recv_data.data(), 0xAA, recv_data.size() * sizeof(T));

    // TEMPORARY DIAGNOSTIC (2026-09-02, KaCCv2 grid-fanout investigation, continued --
    // both isolated reproducers built so far (a plain-Alltoallv call-count loop, and a
    // split/use/free comm-cycling loop matching grid_comm's own lifecycle -- see
    // notes/grid-alltoallv-supermuc-data-loss.md) came back negative on real SuperMUC
    // hardware, as did an even more faithful pre-existing isolated test that replays the
    // real datatype/sizes/grid_comm-per-iteration pattern with a flat, uniform inter-round
    // delay standing in for real computation. A byte-level SEND-vs-RECV comparison across
    // three real crash occurrences (same seed, same call=17) showed the SEND side is
    // always byte-identical and clean, while the RECV side fails non-deterministically --
    // sometimes rank 0, sometimes rank 1, sometimes both -- ruling out a deterministic
    // KaCC-side content bug and pointing at something timing-sensitive in delivery that a
    // FLAT delay never replicates. This captures each rank's real wall-clock arrival time
    // at this exact call, to see whether the corrupting call has unusually large or
    // unusual inter-rank skew compared to clean calls -- something no synthetic reproducer
    // has measured yet, since they were never run against this real skew pattern in the
    // first place. Revert once answered.
    {
      auto const now = std::chrono::high_resolution_clock::now().time_since_epoch();
      auto const us = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
      std::string line = "[grid_alltoallv route_phase SKEW pre] call=" + std::to_string(call_id)
                        + " subrank=" + std::to_string(subcomm.rank()) + "/" + std::to_string(subcomm.size())
                        + " dim_size=" + std::to_string(dim_size) + " is_last=" + std::to_string(is_last ? 1 : 0)
                        + " t_us=" + std::to_string(us) + "\n";
      std::fputs(line.c_str(), stderr);
      std::fflush(stderr);
    }

    // TEMPORARY DIAGNOSTIC (2026-09-02, continued -- the skew capture above showed call
    // #17's inter-rank arrival timing is statistically indistinguishable from clean calls
    // at the same round position (call 3, call 10), directly refuting the timing-race
    // hypothesis on the real crash. This checks two remaining, still-untested angles: (a)
    // whether recv_data's heap address on the corrupting call is suspicious in any way
    // (e.g. coincides with memory very recently freed by the previous iteration's
    // grid_comm teardown -- addresses alone can't prove aliasing, but a suspiciously
    // *repeated* address across iterations, or one close to state.data's own address,
    // would be a real lead), and (b) the allocator's own arena/free-list state
    // (mallinfo2) at this exact point, in case the corrupting call coincides with an
    // allocator-triggered mmap/sbrk transition. Revert once answered.
    {
      auto const recv_addr = reinterpret_cast<std::uintptr_t>(recv_data.data());
      auto const send_addr = reinterpret_cast<std::uintptr_t>(state.data.data());
      std::string line = "[grid_alltoallv route_phase MEM pre] call=" + std::to_string(call_id)
                        + " subrank=" + std::to_string(subcomm.rank()) + "/" + std::to_string(subcomm.size())
                        + " dim_size=" + std::to_string(dim_size) + " is_last=" + std::to_string(is_last ? 1 : 0)
                        + " recv_data.data()=0x" + [&] {
                            char buf[32];
                            std::snprintf(buf, sizeof(buf), "%lx", recv_addr);
                            return std::string{buf};
                          }()
                        + " recv_bytes=" + std::to_string(recv_data.size() * sizeof(T))
                        + " state.data.data()=0x" + [&] {
                            char buf[32];
                            std::snprintf(buf, sizeof(buf), "%lx", send_addr);
                            return std::string{buf};
                          }();
#ifdef __linux__
      auto const mi = mallinfo2();
      line += " arena=" + std::to_string(mi.arena) + " uordblks=" + std::to_string(mi.uordblks)
            + " fordblks=" + std::to_string(mi.fordblks) + " hblkhd=" + std::to_string(mi.hblkhd);
#endif
      line += "\n";
      std::fputs(line.c_str(), stderr);
      std::fflush(stderr);
    }

    // TEMPORARY DIAGNOSTIC/ROOT-CAUSE TEST (2026-09-02, continued -- per explicit user
    // direction to keep root-causing rather than pivot to a workaround yet). Every
    // point-to-point check tried so far (the removed P2P/SHADOW diagnostics) ran AFTER
    // this real exchange had ALREADY completed -- and their mere presence suppressed
    // the bug, so those redos never actually observed a live failure; there is still no
    // MPI_Get_count data from a genuinely failing instance. This makes the REAL exchange
    // itself go through point-to-point instead of the collective, for the 2-rank case
    // (every crash in this investigation has been at subcomm_size==2): if the bug still
    // happens via Isend/Irecv, MPI_Get_count finally tells us what MPI's own bookkeeping
    // says about an exchange that actually fails. If the bug STOPS happening once routed
    // through point-to-point instead of MPI_Alltoallv, that's real evidence the defect is
    // specific to Alltoallv's internal (tuned/pairwise-exchange) algorithm rather than
    // the transport in general -- a genuine root-cause narrowing, and incidentally also a
    // viable permanent fix for this call shape. Only takes this path for `!Ordered`
    // (KaCC's real usage here); `Ordered` mode falls through to the collective call,
    // untouched. Revert or promote to a real fix once answered.
    bool used_p2p_exchange = false;
    if constexpr (!Ordered) {
        if (subcomm_size == 2) {
            used_p2p_exchange   = true;
            int const  peer     = 1 - subcomm.rank();
            constexpr int kExchTag = 11;
            MPI_Request reqs[2];
            MPI_Isend(
                state.data.data() + send_displs[static_cast<std::size_t>(peer)],
                send_counts[static_cast<std::size_t>(peer)],
                dt,
                peer,
                kExchTag,
                subcomm.mpi_handle(),
                &reqs[0]
            );
            MPI_Irecv(recv_data.data(), total_recv, dt, peer, kExchTag, subcomm.mpi_handle(), &reqs[1]);
            MPI_Status statuses[2];
            MPI_Waitall(2, reqs, statuses);
            int actual_count = -1;
            MPI_Get_count(&statuses[1], dt, &actual_count);
            std::string line = "[grid_alltoallv route_phase REALP2P] call=" + std::to_string(call_id)
                              + " subrank=" + std::to_string(subcomm.rank()) + "/" + std::to_string(subcomm.size())
                              + " dim_size=" + std::to_string(dim_size) + " is_last=" + std::to_string(is_last ? 1 : 0)
                              + " expected_count=" + std::to_string(total_recv)
                              + " mpi_get_count=" + std::to_string(actual_count)
                              + " recv_status_error=" + std::to_string(statuses[1].MPI_ERROR) + "\n";
            std::fputs(line.c_str(), stderr);
            std::fflush(stderr);
        }
    }
    if (!used_p2p_exchange) {
        kamping::v2::alltoallv(
            send_data,
            recv_data | views::with_type(dt) | views::with_counts(recv_counts) | views::with_displs(recv_displs),
            subcomm
        );
    }

    // TEMPORARY DIAGNOSTIC (2026-09-02, continued from the "pre" skew capture above): the
    // matching post-call timestamp -- the gap between this rank's pre/post pair is this
    // rank's own call latency; comparing "pre" timestamps for the same call_id across
    // ranks is the actual inter-rank arrival skew this diagnostic exists to measure.
    // Revert once answered.
    {
      auto const now = std::chrono::high_resolution_clock::now().time_since_epoch();
      auto const us = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
      std::string line = "[grid_alltoallv route_phase SKEW post] call=" + std::to_string(call_id)
                        + " subrank=" + std::to_string(subcomm.rank()) + "/" + std::to_string(subcomm.size())
                        + " dim_size=" + std::to_string(dim_size) + " is_last=" + std::to_string(is_last ? 1 : 0)
                        + " t_us=" + std::to_string(us) + "\n";
      std::fputs(line.c_str(), stderr);
      std::fflush(stderr);
    }

    // TEMPORARY DIAGNOSTIC (2026-09-01, KaCCv2 grid-fanout investigation, revised -- see
    // the SEND dump's comment above for why KACC_DIAG_ELEM_BYTES=12 (not sizeof(T)) and
    // why call_id/multiple elements were added) -- matching "SEND" dump above,
    // immediately on the other side of the same alltoallv call, before rebin() (already
    // proven a trivial straight copy for KaCCv2's specific p=2 repro, since
    // next_subtree_size collapses to 1 there) touches recv_data at all. If this dump's
    // bytes already differ from the matching SEND dump's for the same call_id, the
    // corruption is inside this one alltoallv call (or is the poison pattern above,
    // meaning MPI never wrote the slot at all); if they match, rebin/downstream is
    // still implicated despite the p=2 straight-copy argument, and that argument needs
    // re-examining. Revert once answered.
    if (total_recv > 0) {
        constexpr std::size_t KACC_DIAG_RECV_ELEM_BYTES = 12;
        auto const dump_count = std::min<std::size_t>(static_cast<std::size_t>(total_recv), 4);
        std::string line = "[grid_alltoallv route_phase RECV] call=" + std::to_string(call_id)
                          + " subrank=" + std::to_string(subcomm.rank()) + "/" + std::to_string(subcomm.size())
                          + " dim_size=" + std::to_string(dim_size) + " is_last=" + std::to_string(is_last ? 1 : 0)
                          + " total_recv=" + std::to_string(total_recv) + " elems=[";
        for (std::size_t e = 0; e < dump_count; ++e) {
            auto const* elem = reinterpret_cast<unsigned char const*>(&recv_data[e]);
            std::string hex;
            for (std::size_t b = 0; b < KACC_DIAG_RECV_ELEM_BYTES; ++b) {
                char buf[4];
                std::snprintf(buf, sizeof(buf), "%02x", elem[b]);
                hex += buf;
            }
            line += hex;
            if (e + 1 < dump_count) line += ",";
        }
        line += "]\n";
        std::fputs(line.c_str(), stderr);
    }

    // (The bare-MPI_Barrier check that used to live here answered its question and was
    // removed: it crashed 2/2 times, so mere synchronization does NOT suppress the bug --
    // only genuine duplicate data movement did, 4/4. See
    // notes/grid-alltoallv-supermuc-data-loss.md (KaCCv2) for that result. Superseded by
    // the REALP2P swap above, which now tests something sharper: whether point-to-point
    // is inherently safe for this call shape, not just whether extra traffic suppresses
    // a still-present bug.)

    uninit_vector<int> recv_source_rank;
    if constexpr (Ordered) {
        recv_source_rank.resize(static_cast<std::size_t>(total_recv));
        kamping::v2::alltoallv(
            state.source_rank | views::with_counts(send_counts) | views::with_displs(send_displs),
            recv_source_rank | views::with_counts(recv_counts) | views::with_displs(recv_displs),
            subcomm
        );
    }

    // Rebin the received blocks back into remaining-destination-index order (transpose source<->bucket),
    // writing the next tree level and data — and, when ordering by source, the source labels — to state.
    rebin<T, Ordered, ParallelRebin>(
        state,
        recv_data,
        recv_source_rank,
        std::span<int const>{recv_meta},
        std::span<int const>{recv_displs},
        subcomm_size,
        next_subtree_size,
        total_recv
    );
}

} // namespace detail

/// Drop-in grid all-to-all-v (D1). The call site matches the flat KaMPIng v2 alltoallv; only the
/// function name and communicator change — a `grid_comm` instead of a flat comm. The send buffer is a standard variadic
/// `send_buffer_v` (`data | with_counts | with_displs`). The recv buffer constraint depends on the
/// ordering (see `grid_recv_buffer`): `ordered_by_source` needs a variadic recv buffer (`recv_buffer_v`,
/// e.g. `views::auto_recv_v`) to carry the per-source counts; `unordered` accepts any `recv_buffer`
/// (e.g. a plain `std::vector<T>` or `views::resize`).
///
/// @tparam Order Recv ordering tag: `unordered` (default) or `ordered_by_source`.
template <grid_send_buffer SBuf, typename RBuf, is_execution_policy Exec, grid_compatible_output_layout Order = layout::unordered>
    requires grid_recv_buffer<RBuf, Order>
             // ordered_by_source regroups the routed elements into rbuf with a local counting-sort copy
             // (T_send staging -> rbuf), so it needs a host-side conversion; unordered fills rbuf directly
             // via MPI (matched by datatype signature), so no such relation is required.
             && (!std::is_same_v<Order, layout::ordered_by_source>
                 || std::indirectly_copyable<std::ranges::iterator_t<SBuf>, std::ranges::iterator_t<RBuf>>)
auto grid_alltoallv(SBuf&& sbuf, RBuf&& rbuf, grid_comm<Exec> const& grid, [[maybe_unused]] Order order = {})
    -> kamping::v2::result<SBuf, RBuf> {
    using T                 = std::ranges::range_value_t<SBuf>; // send element type; the staging is in T
    constexpr bool ordered  = std::is_same_v<Order, layout::ordered_by_source>;
    constexpr bool parallel = !std::is_same_v<Exec, execution_policy::seq>;

    auto const p  = static_cast<std::size_t>(grid.size());

    if constexpr (kamping::v2::deferred_send_buf_v<std::remove_cvref_t<SBuf>>) {
        sbuf.set_comm_size(static_cast<int>(p));
    }

    auto const dt = mpi::experimental::type(sbuf); // dt_send: drives every staging exchange

    auto&&      send_counts = mpi::experimental::counts(sbuf); // contiguous range of int, size p
    auto&&      send_displs = mpi::experimental::displs(sbuf); // contiguous range of int, size p
    auto const* src         = mpi::experimental::ptr(sbuf);    // T const* (contiguous == &*begin(sbuf))

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
    detail::routing_state<T> state;
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
                std::ranges::copy_n(src + static_cast<std::ptrdiff_t>(send_displs[d]), cnt, state.data.begin() + write);
                write += cnt;
            }
        }
    }
    if constexpr (ordered) {
        state.source_rank.assign(static_cast<std::size_t>(total_send), grid.rank());
    }

    // TEMPORARY DIAGNOSTIC (2026-09-01, KaCCv2 grid-fanout investigation): a monotonic
    // per-process call counter, incremented once per top-level grid_alltoallv() call
    // (shared by every phase inside this one call), so route_phase's SEND/RECV/count
    // dumps can be paired reliably across ranks and rounds instead of by log order
    // alone -- a static-audit subagent flagged the previous log-order-only pairing as
    // the diagnostic's weakest link. Safe as a plain function-local static: this
    // function is only ever called under dstl::execution_policy::seq in KaCCv2's usage
    // (see multistep_impl.hpp), so there is no concurrent-call race to guard against.
    // Revert once answered.
    static long call_id = 0;
    ++call_id;

    // k phases, remote dimension first (dimension 0 → k-1). The last phase sizes rbuf and deposits the
    // routed result straight into it (D8) — there is no separate post-loop sizing or copy.
    auto const num_dims = grid.num_dims();
    for (std::size_t i = 0; i < num_dims; ++i) {
        detail::route_phase<T, ordered, parallel>(
            state,
            grid.dim_size(i),
            dt,
            grid.subcomm(i),
            rbuf,
            /*is_last=*/i + 1 == num_dims,
            static_cast<int>(p),
            call_id
        );
    }

    return kamping::v2::result<SBuf, RBuf>{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
}
} // namespace dstl
