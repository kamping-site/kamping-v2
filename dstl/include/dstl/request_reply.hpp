// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <numeric>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <kamping/types/builtin_types.hpp>
#include <mpi.h>

#include "dstl/default_init_allocator.hpp"
#include "dstl/detail/alltoallv_common.hpp"
#include "dstl/detail/recv_buffer_utils.hpp"
#include "dstl/exchange_layout.hpp"
#include "dstl/tags.hpp"
#include "kamping/v2/collectives/alltoallv.hpp"
#include "kamping/v2/views.hpp"
#include "mpi/buffer.hpp"
#include "mpi/handle.hpp"

/// @file
/// dstl::request_reply — a round-trip collective over a plain MPI_Comm in the style of a std::ranges
/// algorithm: given a buffer of `(value, destination)` request pairs and an output buffer, it ships each
/// request to its destination rank, answers it locally and 1:1 via `make_reply`, and writes the reply for
/// the i-th request into `result[i]`. Built on the reversible `dstl::exchange` (forward + transpose
/// reverse, no reply-direction count negotiation — DSTL-Request-Reply-Design R1/R2). Execution policy
/// `seq` (default) / `par` is a trailing tag that only governs the local loops; the MPI exchange is
/// single-threaded either way. See DSTL-Request-Reply-Flat-Plan.md.

namespace dstl {

namespace detail {

/// The reply element type produced by `make_reply` for a request `Value` — the element-wise result type.
template <typename F, typename Value>
using reply_t = std::remove_cvref_t<std::invoke_result_t<F const&, Value const&>>;

/// Concept: a valid reply functor for a request `Value` — element-wise invocable, producing a
/// trivially-copyable reply (it rides MPI back).
template <typename F, typename Value>
concept reply_fn = std::is_invocable_v<F const&, Value const&> && std::is_trivially_copyable_v<reply_t<F, Value>>;

/// Concept: the shared requirements both request_reply overloads place on their range arguments — a sized
/// random-access request range (the bucketing pack indexes it directly), a contiguous output range whose
/// buffer-ness matches the ordering (`recv_buffer_v` for `ordered_by_source`, plain `recv_buffer`
/// otherwise), a trivially-copyable request value, and a valid reply functor.
template <typename Requests, typename RBuf, typename MakeReply, typename Order>
concept request_reply_args =
    ((std::is_same_v<Order, layout::ordered_by_source> && mpi::experimental::recv_buffer_v<RBuf>)
     || (!std::is_same_v<Order, layout::ordered_by_source> && mpi::experimental::recv_buffer<RBuf>))
    && std::is_trivially_copyable_v<kamping::v2::flat_element_t<Requests>>
    && reply_fn<MakeReply, kamping::v2::flat_element_t<Requests>>;

/// Run `make_reply` over the received requests, 1:1, into a fresh reply vector. The reply buffer is sized
/// to the received-request count (`> p`) and every slot is written, so it uses `uninit_vector` to skip the
/// value-initialization. The loop is OpenMP-parallel when `parallel` is set (via `chunked_for`); reply
/// generation is embarrassingly parallel.
template <typename Reply, typename Value, typename F>
    requires std::is_invocable_r_v<Reply, F const&, Value const&>
uninit_vector<Reply> make_replies(std::span<Value const> recv, F const& make_reply, bool parallel) {
    uninit_vector<Reply> replies(recv.size());
    detail::chunked_for(parallel, static_cast<std::ptrdiff_t>(recv.size()), [&](std::ptrdiff_t lo, std::ptrdiff_t hi) {
        for (std::ptrdiff_t i = lo; i < hi; ++i) {
            replies[static_cast<std::size_t>(i)] = make_reply(recv[static_cast<std::size_t>(i)]);
        }
    });
    return replies;
}

/// The product of `counting_sort_pack`: the `n` elements packed into per-bucket blocks (`data`), plus the
/// per-bucket element counts and their exclusive-scan displacements — the layout a variadic MPI exchange
/// consumes directly. `data` is a `uninit_vector` (every slot is overwritten by the scatter).
template <typename T>
struct packed_buffer {
    uninit_vector<T> data;
    std::vector<int> counts;
    std::vector<int> displs;
};

/// Stable counting-sort pack of `n` elements into per-bucket blocks, in input order: element `i` goes to
/// bucket `key(i)` (in `[0, nbuckets)`) and `value(i)` produces it (stored by value). Returns the packed
/// data alongside the per-bucket `counts` and `displs`.
///
/// The per-chunk histograms the sort builds anyway are summed into `counts` and scanned into `displs`, so
/// there is NO separate (serial) counting pass over the input, and the output buffer is sized from those
/// totals and allocated here — the caller hands in only the keys and a value accessor. Stability holds
/// regardless of OpenMP scheduling: a two-pass scheme (per-chunk histograms, then per-chunk starting
/// cursors offset so chunk `c` follows every earlier chunk for the same bucket) makes each element's slot a
/// pure function of `(chunk, key)` rather than of arrival order — the same shape the grid's ordered deposit
/// uses. `key` must be pure (it is evaluated twice per index, once per pass). Runs serially when `chunks`
/// was built non-parallel (`count() == 1`).
///
/// To avoid false sharing, each chunk's hot count/scatter loop touches only a thread-private vector (its
/// own histogram / cursor copy), never memory shared with a sibling chunk — a single packed `C * nbuckets`
/// buffer would have adjacent chunks' rows share a cache line whenever `nbuckets` is small (the common case
/// here, where `nbuckets` is the comm size). The shared per-chunk arrays are only touched outside the hot
/// loops (one move per chunk, and the serial prefix pass). This mirrors the grid's `reorder_by_source_rank`.
template <typename T, typename KeyFn, typename ValueFn>
packed_buffer<T> counting_sort_pack(chunking const& chunks, int nbuckets, std::ptrdiff_t n, KeyFn key, ValueFn value) {
    int const         C = chunks.count();
    std::size_t const B = static_cast<std::size_t>(nbuckets);

    // Pass 1 — per-chunk histograms. Each chunk counts into a thread-private vector and only stores it
    // (by move) afterwards, so the hot increment loop never shares a cache line with a sibling.
    std::vector<std::vector<int>> hist(static_cast<std::size_t>(C));
    chunks.for_each_chunk([&](int c, std::ptrdiff_t lo, std::ptrdiff_t hi) {
        std::vector<int> local(B, 0);
        for (std::ptrdiff_t i = lo; i < hi; ++i) {
            ++local[static_cast<std::size_t>(key(i))];
        }
        hist[static_cast<std::size_t>(c)] = std::move(local);
    });

    // Per-bucket counts (sum across chunks) and their exclusive-scan displacements — derived from the
    // histograms above, so no separate counting pass. The packed output is sized from `n` and allocated.
    packed_buffer<T> result;
    result.counts.assign(B, 0);
    for (int c = 0; c < C; ++c) {
        for (std::size_t b = 0; b < B; ++b) {
            result.counts[b] += hist[static_cast<std::size_t>(c)][b];
        }
    }
    result.displs.resize(B);
    std::exclusive_scan(result.counts.begin(), result.counts.end(), result.displs.begin(), 0);
    result.data.resize(static_cast<std::size_t>(n));

    // cursor[c][b] = displs[b] + sum over chunks c' < c of hist[c'][b] — the start slot for bucket b in
    // chunk c. Serial O(C * nbuckets); cheap relative to the data passes.
    std::vector<std::vector<int>> cursor(static_cast<std::size_t>(C), std::vector<int>(B));
    for (std::size_t b = 0; b < B; ++b) {
        int acc = result.displs[b];
        for (int c = 0; c < C; ++c) {
            cursor[static_cast<std::size_t>(c)][b] = acc;
            acc += hist[static_cast<std::size_t>(c)][b];
        }
    }

    // Pass 2 — scatter. Each chunk copies its cursor row into a thread-private vector first, so the
    // per-element cursor increments never share a cache line with another chunk.
    chunks.for_each_chunk([&](int c, std::ptrdiff_t lo, std::ptrdiff_t hi) {
        std::vector<int> cur = cursor[static_cast<std::size_t>(c)];
        for (std::ptrdiff_t i = lo; i < hi; ++i) {
            result.data[static_cast<std::size_t>(cur[static_cast<std::size_t>(key(i))]++)] = value(i);
        }
    });
    return result;
}

} // namespace detail

/// Round-trip collective: ship each request to its destination rank, answer it locally and 1:1 via
/// `make_reply`, and deposit the replies grouped by responder rank into `result`.
///
/// The output mirrors the grid alltoallv's recv-ordering model and reuses the same tags. On the flat path
/// the reverse exchange deposits replies already grouped by responder rank (displacements equal to this
/// rank's forward send displacements), so both tags describe the same data layout and differ only in what
/// `result` exposes:
///   * `layout::unordered` (default) — `result` is a plain `recv_buffer`: the replies, grouped by
///     responder, with no per-rank metadata.
///   * `layout::ordered_by_source` — `result` is a variadic `recv_buffer_v` (e.g. `views::auto_recv_v`):
///     the same replies, with the per-responder counts (== this rank's forward send counts) filled in, so
///     the caller can tell which block came from which responder.
///
/// @param requests  A `value_destination_pair_buffer` that is additionally a sized `random_access_range`:
///                   a range of `(value, destination)` pairs indexed directly (and twice) by the bucketing
///                   pack, so a forward-only range must be materialized by the caller first. `value` is the
///                   request payload shipped to `destination` (a rank, plain `int` or a strongly-typed
///                   wrapper); the payload type must be trivially copyable.
/// @param result    The output recv buffer (a contiguous range). For `unordered` any `recv_buffer`
///                   (a plain `std::vector<Reply>` or `vec | views::resize`); for `ordered_by_source` a
///                   `recv_buffer_v` (e.g. `views::auto_recv_v`). Deferred/resizable buffers are sized
///                   automatically; a fixed buffer must already hold this rank's total reply count.
/// @param make_reply  Element-wise `Reply(Value const&)`. The reply must be trivially copyable — it rides
///                    MPI back.
/// @param request_type The MPI datatype of the request value. The pair-range input carries no datatype, so
///                    a non-builtin (custom) value type must supply it here (e.g. from a
///                    `kamping::v2::type_pool`). The builtin overload omits this and deduces it.
/// @param comm        A plain `MPI_Comm` (or anything convertible to one).
/// @tparam Exec       `execution_policy::seq` (default) or `::par`. `par` OpenMP-parallelizes only the local
///                    loops (bucketing pack, `make_reply`); the MPI exchange is single-threaded in both.
///                    `par_comm` is rejected.
/// @tparam Order      `layout::unordered` (default) or `layout::ordered_by_source` (see above).
/// @return The `result` buffer, forwarded (an lvalue is returned by reference, an rvalue by value).
template <
    kamping::v2::value_destination_pair_buffer Requests,
    typename RBuf,
    typename MakeReply,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm  = MPI_Comm,
    is_execution_policy                                    Exec  = execution_policy::seq,
    is_output_layout                                       Order = layout::unordered>
    requires detail::request_reply_args<Requests, RBuf, MakeReply, Order>
auto request_reply(
    Requests&&             requests,
    MPI_Datatype           request_type,
    RBuf&&                 result,
    MakeReply&&            make_reply,
    Comm const&            comm  = MPI_COMM_WORLD,
    [[maybe_unused]] Exec  exec  = {},
    [[maybe_unused]] Order order = {}
) -> RBuf {
    namespace views        = kamping::v2::views;
    using Value            = kamping::v2::flat_element_t<Requests>;
    using Reply            = std::ranges::range_value_t<RBuf>;
    constexpr bool ordered = std::is_same_v<Order, layout::ordered_by_source>;
    static_assert(
        !std::is_same_v<Exec, execution_policy::par_comm>,
        "request_reply: par_comm is not supported by the flat path; use execution_policy::seq or ::par."
    );
    constexpr bool parallel = !std::is_same_v<Exec, execution_policy::seq>;
    MPI_Comm const c        = mpi::experimental::handle(comm);
    int            p        = 0;
    MPI_Comm_size(c, &p);

    // Bucket the requests by destination with a stable counting-sort pack, which also yields the
    // per-destination counts/displs (no separate serial counting pass) and allocates the packed buffer.
    //
    // The pack reads each request twice (the histogram pass and the scatter pass) and indexes by position,
    // so it indexes `requests` directly — no upfront copy. This is why the input is constrained to a sized
    // `random_access_range` above (beyond the `forward_range` that `value_destination_pair_buffer` alone
    // guarantees): materializing a forward-only range into random-access arrays is a cost left to the
    // caller. The cheap destination rank is recomputed each pass rather than cached.
    auto const n     = static_cast<std::ptrdiff_t>(std::ranges::size(requests));
    auto       first = std::ranges::begin(requests);
    auto const pack  = detail::counting_sort_pack<Value>(
        detail::chunking{parallel, n},
        p,
        n,
        [&](std::ptrdiff_t i) { return mpi::experimental::to_rank(std::get<1>(first[i])); },
        [&](std::ptrdiff_t i) -> Value { return std::get<0>(first[i]); }
    );

    // Forward exchange: a kamping::v2::alltoallv into an auto-managed recv buffer (which negotiates the
    // per-source recv counts and sizes recv_requests). `request_type` is attached to both buffers (matched
    // by signature, like a plain MPI_Alltoallv). Snapshot the negotiated layout off the two buffers so the
    // return trip can reuse it.
    detail::uninit_vector<Value> recv_requests;
    auto                         fwd = kamping::v2::alltoallv(
        pack.data | views::with_type(request_type) | views::with_counts(pack.counts) | views::with_displs(pack.displs),
        recv_requests | views::with_type(request_type) | views::auto_recv_v,
        c
    );
    exchange_layout const layout{fwd.send, fwd.recv};

    // Answer each received request locally, 1:1, in forward recv order.
    std::span<Value const> const recv{recv_requests.data(), recv_requests.size()};
    auto                         replies = detail::make_replies<Reply, Value>(recv, make_reply, parallel);

    // Transpose return: ship the replies back along the inverse route. The reversed layout's send side
    // groups the replies as the forward recv layout, so they land grouped by responder rank (displs ==
    // forward send_displs). The reply MPI datatype is taken from the output buffer (the recv buffer carries
    // its own type) and attached to the reply send data.
    exchange_layout const rev        = reverse_layout(layout);
    MPI_Datatype const    reply_dt   = mpi::experimental::type(result);
    auto                  reply_send = replies | views::with_type(reply_dt) | views::with_counts(rev.send_counts)
                                       | views::with_displs(rev.send_displs);
    if constexpr (ordered) {
        // result is a managed recv_buffer_v: kamping::v2::alltoallv fills its per-responder counts (== this
        // rank's forward send counts) and sizes it. Borrow result here (it is returned below).
        kamping::v2::alltoallv(reply_send, result, c);
    } else {
        // result is a plain recv_buffer: size it (opt-in resize) and attach the reversed recv layout, so the
        // alltoallv deposits without re-negotiating counts.
        detail::ensure_recv_capacity(result, rev.recv_total());
        kamping::v2::alltoallv(
            reply_send,
            result | views::with_counts(rev.recv_counts) | views::with_displs(rev.recv_displs),
            c
        );
    }

    return std::forward<RBuf>(result);
}

/// Overload for a request value type the buffer protocol can deduce a datatype for — i.e. a builtin MPI
/// type (e.g. `int`). Deduces the request datatype and forwards to the explicit-datatype overload above. A
/// non-builtin (custom) request value does not match this overload, so the caller must use the
/// `MPI_Datatype` overload and supply the datatype (e.g. from a `kamping::v2::type_pool`).
template <
    kamping::v2::value_destination_pair_buffer Requests,
    typename RBuf,
    typename MakeReply,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm  = MPI_Comm,
    is_execution_policy                                    Exec  = execution_policy::seq,
    is_output_layout                                       Order = layout::unordered>
    requires detail::request_reply_args<Requests, RBuf, MakeReply, Order>
             && kamping::types::is_builtin_type_v<kamping::v2::flat_element_t<Requests>>
auto request_reply(
    Requests&&  requests,
    RBuf&&      result,
    MakeReply&& make_reply,
    Comm const& comm  = MPI_COMM_WORLD,
    Exec        exec  = {},
    Order       order = {}
) -> RBuf {
    return request_reply(
        std::forward<Requests>(requests),
        kamping::types::builtin_type<kamping::v2::flat_element_t<Requests>>::data_type(),
        std::forward<RBuf>(result),
        std::forward<MakeReply>(make_reply),
        comm,
        exec,
        order
    );
}

} // namespace dstl
