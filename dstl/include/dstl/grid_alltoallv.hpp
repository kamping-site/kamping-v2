// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
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
    [[maybe_unused]] int         global_size
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

    // recv_counts / recv_displs were derived locally from recv_meta above (the BEFORE half of the
    // phase diagram at the top of this function), so we attach them explicitly rather than let the
    // kamping::v2 layer re-negotiate them (a redundant collective per phase).
    // with_type(dt) preserves the caller's MPI datatype.
    auto send_data =
        state.data | views::with_type(dt) | views::with_counts(send_counts) | views::with_displs(send_displs);

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
    kamping::v2::alltoallv(
        send_data,
        recv_data | views::with_type(dt) | views::with_counts(recv_counts) | views::with_displs(recv_displs),
        subcomm
    );
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
            static_cast<int>(p)
        );
    }

    return kamping::v2::result<SBuf, RBuf>{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
}
} // namespace dstl
