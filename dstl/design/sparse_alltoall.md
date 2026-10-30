# Design: dstl sparse alltoall

Sparse personalized all-to-all (dynamic sparse data exchange): every rank sends messages to a
(small, dynamic) set of destinations, and no rank knows in advance from whom it will receive.
Implements the NBX, PCX and RSX protocols of Hoefler, Siebert, Lumsdaine, *Scalable Communication
Protocols for Dynamic Sparse Data Exchange* (PPoPP 2010), selected by a tag. Unlike
`MPI_Alltoallv`, NBX has communication complexity linear in the number of messages actually sent,
not in the communicator size.

Implementation: `dstl/include/dstl/sparse_alltoall.hpp`; tests: `dstl/tests/sparse_alltoall_test.cpp`.

---

## Building blocks

| Building block | Where |
|---|---|
| `probe_result`, `v2::mprobe`, `v2::improbe`; move-only `message` | #83 |
| `ibarrier` (v2 returns a `request`) | #84 |
| `v2::mrecv`, `v2::imrecv` (take `probe_result` by value, size deferred buffers via `infer(comm_op::mrecv, ...)`) | #85 |
| `win`, `win_view`, `win_allocate` | #86 |
| `cache<Kind, T>` / `keyval<Kind, T>` (per-communicator state, see `docs/attribute_caching.md`) | #87 |
| Message concepts: structure (`nested_send_buffer`, `sparse_nested_send_buffer`) vs. capability (`range_message`, `buffer_message`) | #88 |

---

## Algorithm selection

```cpp
namespace dstl::sparse_alltoall_algorithm {
struct nbx_t {}; inline constexpr nbx_t nbx{}; // nonblocking consensus: issend + improbe + ibarrier (default)
struct pcx_t {}; inline constexpr pcx_t pcx{}; // personalized census: reduce_scatter of message counts
struct rsx_t {}; inline constexpr rsx_t rsx{}; // remote summing: MPI_Accumulate into a counter window
}
template <typename T>
concept is_sparse_alltoall_algorithm = /* one of the tag types */;
```

The tag is the first argument (like `std::execution` policies); the overload without it uses NBX.

**No PEX.** PEX (an `alltoall` of per-destination counts, then the data) is what
`kamping::v2::alltoallv` with `auto_counts` / `auto_displs` already does. PEX sends the data
point-to-point instead of with `MPI_Alltoallv`, but both pay O(p) for the count exchange, so there
is no asymptotic benefit. For dense patterns, use `alltoallv`.

---

## Layer 1 — callback primitive

```cpp
template <sparse_alltoall_send_buffer SBuf, std::invocable<kamping::v2::probe_result> OnMessage, tag Tag, Comm>
void dstl::sparse_alltoall(Algorithm, SBuf&& sbuf, OnMessage&& on_message, Tag tag, Comm const& comm = MPI_COMM_WORLD);

// RSX with a reusable counter window per communicator:
void dstl::sparse_alltoall(rsx, SBuf&&, OnMessage&&, Tag, Comm const&, dstl::rsx_window_cache& windows);
```

### Input: `sparse_alltoall_send_buffer`

Two forms, both built on the message concepts of `payload.hpp`:

- **sparse**: a range of `(destination, message)` pairs (`sparse_nested_send_buffer`); a destination
  may appear more than once;
- **dense**: a range of messages indexed by destination (`nested_send_buffer`); at most one entry per
  process, fewer entries are fine. Scanning it takes O(p) local time.

Messages are sent in place, so they must be **`buffer_message`s** (pointer + count via the buffer
protocol): contiguous ranges, and also buffers adapted via `buffer_traits` or `mpi_ptr()` /
`mpi_count()` that are not ranges. Empty messages are not sent.

**Lifetime.** Messages are sent nonblocking while the input is iterated, so their data must outlive
the iteration that posts them (`messages_outlive_iteration`): either the input yields references
into storage that outlives the call, or the messages do not own their data (`std::span`). Messages
produced by value that own their data (e.g. `std::views::transform` returning `std::vector`s) are
rejected with a `static_assert`. Staging them instead is an open point (below).

### Datatype: the value-type channel

The datatype is attached once to the whole input, not per message:

```cpp
dstl::sparse_alltoall(sends | kamping::v2::views::with_value_pool(p), on_message, tag, comm);
dstl::sparse_alltoall(sends | kamping::v2::views::with_value_type(dt), on_message, tag, comm);
dstl::sparse_alltoall(sends, on_message, tag, comm); // builtin payload: value_type() fallback
```

`kamping::v2::value_type(sbuf)` is read once and used for every send: one datatype per call, which
must outlive the call (guaranteed by a `type_pool`). Messages need no datatype of their own.

### Callback: `probe_result`

All three protocols learn about incoming messages by probing with `MPI_ANY_SOURCE`, so the callback
receives the `probe_result` directly and receives it with `kamping::v2::mrecv` / `imrecv`, which
size deferred buffers (`views::resize`) from the probed count with the *receive* buffer's datatype.
A message-handle abstraction (`incoming_message`) was only needed to hide PEX's "size already
known" case and was dropped with PEX.

Contract:

- The callback must **start the receive before it returns** (`mrecv`, or `imrecv` for asynchronous
  receives). Deferring receives can deadlock: NBX's synchronous sends might only complete once the
  receive has started, and PCX/RSX wait for their own sends before returning.
- It must not block on anything outside MPI that depends on other processes.
- Messages from the same source arrive in send order (MPI non-overtaking).
- Dropping a `probe_result` without receiving it terminates (`message` is move-only and checked).

### Tags and isolation across calls

All messages of a call are sent and probed with `tag` on `comm`. **Decision:** isolation is the
caller's responsibility, as for MPI point-to-point. Consecutive calls on the same communicator must
use different tags (e.g. alternating) or be separated by a synchronization, and no other messages
may use `tag` on `comm` during the call. Otherwise a fast process's next call can send messages
that a slow process still probing in the previous call matches. The tag has no default.

The contract is easy to break: the first version of the tests used tag 0 for every test and failed
only under load. Isolation inside the library is an open point (below).

---

## Shared send phase

All protocols send every non-empty message from one place (`detail::post_sends`), parameterized by
the send mode and a per-message hook:

- NBX uses **`issend`**, PCX and RSX use `isend`;
- PCX counts messages per destination, RSX accumulates into the counter window;
- requests go into a `request_vector`; each message is sent as `message | with_type(value_type)`.

---

## Protocols

### NBX

```
1. issend every message (synchronous mode!), collecting requests
2. loop:
       if (auto probe = improbe(MPI_ANY_SOURCE, tag, comm)) on_message(std::move(*probe))
       if (!barrier_started && testall(sends)) { barrier = ibarrier(comm); barrier_started = true; }
       if (barrier_started && barrier.test()) return
```

The sends **must be synchronous**: termination relies on a completed send meaning the receiver has
matched the message, which is what allows a process to enter the `ibarrier` once all its sends are
complete. With `isend`, a small message can complete locally before it is matched, and the
protocol can terminate with messages still in flight.

### PCX

Message counts per destination (counts, not 0/1 flags, since a destination may repeat) are
reduce-scattered (`MPI_Reduce_scatter_block` into a separate `int`, not in place) to give the
number k of incoming messages; then `mprobe(MPI_ANY_SOURCE, tag)` k times, and wait for the own
sends. No explicit progress is needed: a process waiting is always inside MPI (the reduce-scatter,
`mprobe`, `mrecv`, `waitall`), which also progresses its rendezvous sends. O(p) memory and
communication per process.

### RSX

Each sender `MPI_Accumulate`s +1 into a counter on each destination within a fence epoch; then probe
as in PCX. The counter window (`detail::rsx_counter`):

- one `int64` per process, allocated with `win_allocate` and info hints (`no_locks`,
  `accumulate_ordering = none`, `accumulate_ops = same_op`, `same_size`, `same_disp_unit`);
- the counter only grows and each exchange counts the difference to the previous one, so there are
  no local stores after construction: the fences assert `NOPRECEDE | NOSTORE` and
  `NOSUCCEED | NOSTORE`;
- allocating and freeing a window is collective and expensive, so it can be reused:
  `dstl::rsx_window_cache` is a `cache<attribute::comm, rsx_counter>` holding one window per
  communicator. The overload without a cache uses a temporary one (a window per call).

---

## Open points

### Staging owning messages

Messages produced by value that own their data (e.g. `views::serialize` via `std::views::transform`)
are currently rejected. The shared send phase could instead move each one into an internal vector
that lives until all sends complete; for inputs satisfying `messages_outlive_iteration` this would
compile away. Until then, materialize first:

```cpp
auto staged = sends | std::views::transform(/* ... serialize ... */) | std::ranges::to<std::vector>();
dstl::sparse_alltoall(staged, callback, tag, comm);
```

### Isolation inside the library

Instead of the tag contract, cache a private duplicate of each user communicator
(`cache<attribute::comm, comm>`) and communicate on it, like MPI collectives do internally. Messages
of different calls would still need separating (alternating internal tags, or a sequence number per
cached communicator), but no longer user traffic.

### Layer 2 — flat buffer convenience

Takes a variadic send buffer (flat data + per-rank counts + displacements), sends only to ranks
with non-zero counts, and writes into a recv buffer:

```cpp
void dstl::sparse_alltoall(Algorithm, SBuf&& sbuf, RBuf&& rbuf, Layout = layout::unordered{}, Tag tag, Comm const& comm);
```

- `layout::unordered` (default): arrival order, into a plain `recv_buffer`;
- `layout::ordered_by_source`: the `MPI_Alltoallv` layout, into a `recv_buffer_v` (O(p) counts, as
  in `request_reply`; `detail::size_from_source_counts` sizes it). Offsets depend on all incoming
  sizes:
  - NBX cannot hold probes until the end (see the callback contract), so it receives into a staging
    buffer in arrival order and copies segments into source order afterwards (stable, so repeated
    sources keep send order);
  - PCX/RSX know k up front: `mprobe` all k messages, size the output from their counts, then `mrecv`
    each directly at its offset — zero-copy. This cannot go through the Layer 1 callback (it would
    hold probes across the end of the call), so it is implemented per protocol.

### Convenience variants

Built on Layer 1, collecting all received messages (e.g. as `(source, data)` pairs, or flat data plus
per-element sources), accepting the algorithm tag like it.

---

## Dropped

- **`transform_messages`**: was mainly for attaching a datatype per message; the value-type channel
  (`with_value_type` / `with_value_pool`, see [`value_type.md`](value_type.md)) replaces that with a
  single annotation. Other per-message transformations are written with `std::views::transform`.
- **`incoming_message`** and **PEX**: see "Callback" and "Algorithm selection".
