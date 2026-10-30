// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

/// @file
/// @brief Sparse personalized all-to-all (dynamic sparse data exchange) with the NBX, PCX and RSX protocols of
/// Hoefler, Siebert and Lumsdaine, "Scalable Communication Protocols for Dynamic Sparse Data Exchange" (PPoPP 2010).

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include "kamping/v2/collectives/ibarrier.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/completion.hpp"
#include "kamping/v2/kassert.hpp"
#include "kamping/v2/p2p/isend.hpp"
#include "kamping/v2/p2p/probe.hpp"
#include "kamping/v2/p2p/send_mode.hpp"
#include "kamping/v2/request.hpp"
#include "kamping/v2/views/payload.hpp"
#include "kamping/v2/views/with_type_view.hpp"
#include "mpi/cache.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"
#include "mpi/info.hpp"
#include "mpi/win.hpp"

namespace dstl {

/// @brief Tags selecting the protocol of @ref sparse_alltoall.
namespace sparse_alltoall_algorithm {
/// Nonblocking consensus: synchronous sends, probing until a nonblocking barrier completes. No collective on the
/// message counts; the default.
struct nbx_t {};
inline constexpr nbx_t nbx{};

/// Personalized census: a reduce-scatter of per-destination message counts tells each process how many messages to
/// probe for.
struct pcx_t {};
inline constexpr pcx_t pcx{};

/// Remote summing: each sender adds to a counter window on its destinations, which then know how many messages to
/// probe for.
struct rsx_t {};
inline constexpr rsx_t rsx{};
} // namespace sparse_alltoall_algorithm

/// @brief One of the tags in @ref sparse_alltoall_algorithm.
template <typename T>
concept is_sparse_alltoall_algorithm =
    std::same_as<T, sparse_alltoall_algorithm::nbx_t> || std::same_as<T, sparse_alltoall_algorithm::pcx_t>
    || std::same_as<T, sparse_alltoall_algorithm::rsx_t>;

/// @brief Input of @ref sparse_alltoall, in one of two forms:
/// - **sparse**: a range of `(destination, message)` pairs (`kamping::v2::sparse_nested_send_buffer`), or
/// - **dense**: a range of messages whose index is the destination (`kamping::v2::nested_send_buffer`).
///
/// Messages are sent in place, so they must be accessible through the buffer protocol
/// (`kamping::v2::buffer_message`: a pointer and a count). Contiguous ranges qualify, as do custom buffers adapted via
/// `buffer_traits` or `mpi_ptr()` / `mpi_count()`. Messages need no datatype of their own: it comes from the
/// value-type channel (`kamping::v2::value_type`), one datatype for all messages.
template <typename SBuf>
concept sparse_alltoall_send_buffer =
    (kamping::v2::nested_send_buffer<SBuf> || kamping::v2::sparse_nested_send_buffer<SBuf>)
    && kamping::v2::buffer_message<kamping::v2::nested_message_t<SBuf>>;

namespace detail {

/// Counter window of the RSX protocol: one `int64` per process, counting the messages addressed to it.
///
/// The counter only ever grows, so each exchange counts the difference to the previous one. Without local stores
/// after construction, every exchange can assert `MPI_MODE_NOSTORE` on both of its fences, and a window can be reused
/// across exchanges (see @ref rsx_window_cache).
class rsx_counter {
public:
    /// Collective: allocates the window on `comm`.
    explicit rsx_counter(MPI_Comm comm) {
        // Optimization hints; implementations may ignore any of them.
        mpi::experimental::info info;
        info.set("no_locks", true);              // only fences, no passive-target locks
        info.set("accumulate_ordering", "none"); // the increments commute, no ordering needed
        info.set("accumulate_ops", "same_op");   // every accumulate is an MPI_SUM
        info.set("same_size", true);             // every process exposes a single counter
        info.set("same_disp_unit", true);        // with the same displacement unit
        auto [w, local] = mpi::experimental::win_allocate<std::int64_t>(1, comm, info);
        _win            = std::move(w);
        _count          = local.data();
        *_count         = 0;
        // Makes the initial store visible; opens no epoch, so the first exchange can assert MPI_MODE_NOPRECEDE.
        _win.fence(MPI_MODE_NOPRECEDE | MPI_MODE_NOSUCCEED);
    }

    /// Collective: opens the counting epoch.
    void start() {
        _win.fence(MPI_MODE_NOPRECEDE | MPI_MODE_NOSTORE);
    }

    /// Counts one message to `dest` (within the epoch).
    void add_message(int dest) {
        std::int64_t const one = 1;
        int                err =
            MPI_Accumulate(&one, 1, MPI_INT64_T, dest, 0, 1, MPI_INT64_T, MPI_SUM, mpi::experimental::handle(_win));
        if (err != MPI_SUCCESS) {
            throw mpi::experimental::mpi_error(err);
        }
    }

    /// Collective: closes the epoch and returns the number of messages addressed to this process since @ref start.
    [[nodiscard]] std::int64_t finish() {
        _win.fence(MPI_MODE_NOSUCCEED | MPI_MODE_NOSTORE);
        std::int64_t const total = *_count;
        std::int64_t const n     = total - _last;
        _last                    = total;
        return n;
    }

private:
    mpi::experimental::win _win;
    std::int64_t*          _count = nullptr;
    std::int64_t           _last  = 0;
};

/// Posts a nonblocking send of every non-empty message in `sbuf` with `mode`, collecting the requests in `requests`.
/// `on_send(dest)` is called once per posted message, before its send. All messages use the value type of `sbuf`.
template <sparse_alltoall_send_buffer SBuf, kamping::v2::is_send_mode SendMode, typename Comm, typename OnSend>
void post_sends(
    SBuf const& sbuf, SendMode mode, int tag, Comm const& comm, kamping::v2::request_vector& requests, OnSend&& on_send
) {
    MPI_Datatype const type = kamping::v2::value_type(sbuf);
    auto               send = [&](int dest, auto const& message) {
        if (mpi::experimental::count(message) == 0) {
            return;
        }
        on_send(dest);
        kamping::v2::isend(
            mode,
            requests.emplace_back(),
            message | kamping::v2::views::with_type(type),
            dest,
            tag,
            comm
        );
    };
    if constexpr (kamping::v2::sparse_nested_send_buffer<SBuf>) {
        for (auto const& [dest, message]: sbuf) {
            send(mpi::experimental::to_rank(dest), message);
        }
    } else {
        int dest = 0;
        for (auto const& message: sbuf) {
            send(dest++, message);
        }
    }
}

/// PCX and RSX: receive exactly `num_recvs` messages, then wait for this process's own sends.
template <typename OnMessage, typename Comm>
void receive_counted(
    std::int64_t num_recvs, OnMessage& on_message, int tag, Comm const& comm, kamping::v2::request_vector& sends
) {
    for (std::int64_t i = 0; i < num_recvs; ++i) {
        std::invoke(on_message, kamping::v2::mprobe(MPI_ANY_SOURCE, tag, comm));
    }
    kamping::v2::waitall(sends);
}

} // namespace detail

/// @brief Reusable RSX counter windows, one per communicator.
///
/// Pass one to the RSX overload of @ref sparse_alltoall to allocate the counter window only on the first exchange on
/// each communicator, instead of on every exchange. Destroying the cache frees the windows (collective). See
/// `docs/attribute_caching.md`.
using rsx_window_cache = mpi::experimental::cache<mpi::experimental::attribute::comm, detail::rsx_counter>;

/// @brief Sparse personalized all-to-all with the NBX protocol.
///
/// Sends every non-empty message in `sbuf` and calls `on_message` with a `probe_result` for every message received.
/// Only processes that exchange messages communicate point-to-point; completion is detected with a nonblocking
/// barrier, so there is no collective on the message counts.
///
/// **Input** (see @ref sparse_alltoall_send_buffer). `sbuf` is either a range of `(destination, message)` pairs, in
/// which a destination may appear more than once, or a range of messages indexed by destination (at most one entry
/// per process). Each message is a buffer-protocol message (e.g. a contiguous range), sent in place; empty messages
/// are not sent, so only processes that exchange
/// data communicate. The MPI datatype of the elements is taken from the value-type channel: attach it with
/// `views::with_value_type(dt)` or `views::with_value_pool(pool)`, or rely on the builtin fallback. The dense form
/// takes O(p) local time to scan; the sparse form only touches the messages actually sent.
///
/// **Callback.** `on_message` must start receiving the message (`kamping::v2::mrecv` / `imrecv`) before it returns,
/// and must not block on anything outside MPI that depends on other processes. Messages from the same source arrive
/// in the order they were sent.
///
/// **Tags.** All messages are sent and probed with `tag` on `comm`. Consecutive calls on the same communicator must
/// use different tags, or be separated by a synchronization; and no other messages may use `tag` on `comm` during the
/// call.
///
/// @throws mpi_error if an MPI call fails.
template <
    sparse_alltoall_send_buffer                            SBuf,
    std::invocable<kamping::v2::probe_result>              OnMessage,
    mpi::experimental::tag                                 Tag  = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void sparse_alltoall(
    sparse_alltoall_algorithm::nbx_t, SBuf&& sbuf, OnMessage&& on_message, Tag tag, Comm const& comm = MPI_COMM_WORLD
) {
    int const                   t = mpi::experimental::to_tag(tag);
    kamping::v2::request_vector sends;
    // Synchronous sends: a completed send means the receiver has matched it, which is what makes it safe to enter
    // the barrier once all own sends are complete.
    detail::post_sends(sbuf, kamping::v2::send_mode::sync, t, comm, sends, [](int) {});

    mpi::experimental::request barrier;
    bool                       barrier_started = false;
    while (true) {
        if (auto probe = kamping::v2::improbe(MPI_ANY_SOURCE, t, comm)) {
            std::invoke(on_message, std::move(*probe));
        }
        if (barrier_started) {
            if (barrier.test()) {
                return;
            }
        } else if (kamping::v2::testall(sends)) {
            barrier         = kamping::v2::ibarrier(comm);
            barrier_started = true;
        }
    }
}

/// @brief Sparse personalized all-to-all with the PCX protocol.
///
/// Like the NBX overload, but each process learns the number of messages addressed to it from a reduce-scatter of
/// per-destination message counts (O(p) memory and communication per process), then probes for exactly that many.
///
/// @throws mpi_error if an MPI call fails.
template <
    sparse_alltoall_send_buffer                            SBuf,
    std::invocable<kamping::v2::probe_result>              OnMessage,
    mpi::experimental::tag                                 Tag  = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void sparse_alltoall(
    sparse_alltoall_algorithm::pcx_t, SBuf&& sbuf, OnMessage&& on_message, Tag tag, Comm const& comm = MPI_COMM_WORLD
) {
    int const                   t = mpi::experimental::to_tag(tag);
    MPI_Comm const              c = mpi::experimental::handle(comm);
    std::size_t const           p = static_cast<std::size_t>(kamping::v2::comm_view{c}.size());
    std::vector<int>            counts(p, 0);
    kamping::v2::request_vector sends;
    detail::post_sends(sbuf, kamping::v2::send_mode::standard, t, comm, sends, [&](int dest) {
        KAMPING_V2_ASSERT(dest >= 0 && static_cast<std::size_t>(dest) < p, "destination rank out of range");
        ++counts[static_cast<std::size_t>(dest)];
    });

    int num_recvs = 0;
    int err       = MPI_Reduce_scatter_block(counts.data(), &num_recvs, 1, MPI_INT, MPI_SUM, c);
    if (err != MPI_SUCCESS) {
        throw mpi::experimental::mpi_error(err);
    }
    detail::receive_counted(num_recvs, on_message, t, comm, sends);
}

/// @brief Sparse personalized all-to-all with the RSX protocol, using cached counter windows.
///
/// Like the NBX overload, but each sender adds to a counter window on each of its destinations, which then probe for
/// exactly as many messages as were counted. The counter window for `comm` is taken from `windows`, and allocated
/// there on the first exchange on `comm`.
///
/// @throws mpi_error if an MPI call fails.
template <
    sparse_alltoall_send_buffer                            SBuf,
    std::invocable<kamping::v2::probe_result>              OnMessage,
    mpi::experimental::tag                                 Tag  = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void sparse_alltoall(
    sparse_alltoall_algorithm::rsx_t,
    SBuf&&            sbuf,
    OnMessage&&       on_message,
    Tag               tag,
    Comm const&       comm,
    rsx_window_cache& windows
) {
    int const      t       = mpi::experimental::to_tag(tag);
    MPI_Comm const c       = mpi::experimental::handle(comm);
    auto&          counter = windows.get_or_create(c, [c] { return detail::rsx_counter(c); });

    kamping::v2::request_vector sends;
    counter.start();
    detail::post_sends(sbuf, kamping::v2::send_mode::standard, t, comm, sends, [&](int dest) {
        counter.add_message(dest);
    });
    detail::receive_counted(counter.finish(), on_message, t, comm, sends);
}

/// @brief Sparse personalized all-to-all with the RSX protocol.
///
/// Like the cached RSX overload, but allocates and frees a counter window on every call (both collective). Prefer
/// the cached overload for repeated exchanges on the same communicator.
///
/// @throws mpi_error if an MPI call fails.
template <
    sparse_alltoall_send_buffer                            SBuf,
    std::invocable<kamping::v2::probe_result>              OnMessage,
    mpi::experimental::tag                                 Tag  = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void sparse_alltoall(
    sparse_alltoall_algorithm::rsx_t, SBuf&& sbuf, OnMessage&& on_message, Tag tag, Comm const& comm = MPI_COMM_WORLD
) {
    rsx_window_cache windows;
    sparse_alltoall(
        sparse_alltoall_algorithm::rsx,
        std::forward<SBuf>(sbuf),
        std::forward<OnMessage>(on_message),
        std::move(tag),
        comm,
        windows
    );
}

/// @brief Sparse personalized all-to-all with the default protocol (NBX); see the NBX overload.
template <
    sparse_alltoall_send_buffer                            SBuf,
    std::invocable<kamping::v2::probe_result>              OnMessage,
    mpi::experimental::tag                                 Tag  = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void sparse_alltoall(SBuf&& sbuf, OnMessage&& on_message, Tag tag, Comm const& comm = MPI_COMM_WORLD) {
    sparse_alltoall(
        sparse_alltoall_algorithm::nbx,
        std::forward<SBuf>(sbuf),
        std::forward<OnMessage>(on_message),
        std::move(tag),
        comm
    );
}

} // namespace dstl
