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

### Layer 2 — receive into one buffer

Receives all messages into a single recv buffer, in arrival order:

```cpp
std::size_t dstl::sparse_alltoall(Algorithm, SBuf&& sbuf, RBuf&& rbuf, Tag tag, Comm const& comm);
```

**Input: no separate interface.** Any Layer 1 input works. A flat variadic send buffer (data + per-rank
counts + displacements) becomes one through a `chunked` view, the inverse of `flatten_v`: a dense
range of `std::span`s, one per rank, into the flat data. The spans do not own their data (so they
pass `messages_outlive_iteration`), empty ones are not sent, and the view's value type is the flat
buffer's `type()`:

```cpp
dstl::sparse_alltoall(nbx, flat | views::with_counts(counts) | views::auto_displs() | views::chunked, rbuf, tag, comm);
```

`chunked` is not marked `already_flat_buffer`; `flatten_v(chunked(x))` round-trips. It needs an element
type for its spans, so the flat buffer must expose one (`range_value_t` or `value_type`).

**Thin layer on Layer 1.** Layer 2 is a Layer 1 call with a callback that receives each message at
the next free position of the recv buffer; the protocols are unchanged. All arguments after the recv
buffer (tag, communicator, cached RSX windows) are forwarded to Layer 1. It returns the number of
elements received.

**Output: arrival order only.**

- **Presized buffer** (the default, as everywhere in kamping): each message is received with `imrecv`
  at the running offset right after it is probed, so receives overlap; an `imrecv` counts as a started
  receive, so NBX's synchronous sends complete as with `mrecv`. All receives are waited for after the
  exchange. The buffer must be large enough (asserted).
- **`views::resize`**: the total is unknown until the exchange ends, and the buffer must not be resized
  under in-flight receives. Each message is therefore received with a blocking `mrecv`, after growing
  the buffer to fit it (amortized by the container's geometric growth). Overlapping receives would
  need batches of probes and a batch-aware Layer 1 (an earlier draft did this); a matched message is
  already on its way, so `mrecv` mostly waits for its data and the gain is small. At the end the
  buffer is sized to exactly the received elements. This relies on the resize contract below.

**Resize contract.** `resize_for_receive(n)` (and `mpi_resize_for_receive`) resize to `n` elements,
keeping the values of `[0, min(old, n))`; new elements are unspecified. Like a receive, a resize only
touches the range it has to. Standard containers (`resize`) and the Kokkos adapter (`Kokkos::resize`)
already behave this way; the contract is documented and tested separately, before Layer 2.

**No provenance.** Arrival order loses which data came from where. Layer 2 does not record it: a
caller who needs sources, per-message counts or any other control uses the Layer 1 callback directly.

### No `ordered_by_source`

An `MPI_Alltoallv` layout (grouped by source, counts and displacements per source) needs the counts
per source. They are either given or exchanged with an `alltoall` (as `infer(comm_op::alltoallv)` does
for `auto_counts`). Once they are known, every process knows exactly whom it receives from, so the
protocols are pointless: post an `irecv` per non-zero source at its displacement, send, wait — PEX,
i.e. an `alltoallv` that communicates point-to-point only where there is data. That is an optimization
of `alltoallv`, not of `sparse_alltoall`: if wanted, it belongs in dstl's alltoallv family (next to
`flat_alltoallv` and `grid_alltoallv`). The protocols only pay off without counts, i.e. in arrival
order.

Producing the `MPI_Alltoallv` layout from arrival order plus per-message sources (recorded with a
Layer 1 callback; stable sort by source, permute the segments) is a local operation and applies to
any unordered data.

### Convenience variants

Built on Layer 1, collecting all received messages (e.g. as `(source, data)` pairs, or flat data plus
per-element sources), accepting the algorithm tag like it. Layer 2 covers flat data without sources.

---

## Dropped

- **`transform_messages`**: was mainly for attaching a datatype per message; the value-type channel
  (`with_value_type` / `with_value_pool`, see [`value_type.md`](value_type.md)) replaces that with a
  single annotation. Other per-message transformations are written with `std::views::transform`.
- **`incoming_message`** and **PEX**: see "Callback" and "Algorithm selection".
