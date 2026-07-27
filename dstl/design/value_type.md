# Design: `with_value_type` and the value-type channel

Motivation: structured send buffers — a range of `(value, rank)` pairs
(`value_destination_pair`), a range of `(rank, inner_range)` pairs
(`sparse_nested_send_buffer`), or a plain range-of-ranges
(`nested_send_buffer`) — carry a **payload element** that is distinct from the
range's own `value_type`. Two upcoming consumers need to attach an explicit
MPI datatype to that payload:

- `dstl::request_reply`, which takes a `value_destination_pair` buffer
  **natively** (no packing into a `send_buf_v` first) and must send each value
  with a caller-chosen MPI type.
- `dstl::sparse_alltoall` (see [`sparse_alltoall.md`](sparse_alltoall.md)),
  replacing the per-message `transform_messages(with_type(...))` construct with
  a single whole-buffer annotation.

---

## Why not reuse `with_type`

`with_type` sets `mpi_type()` — "the MPI type of *this buffer's elements*". For
a structured buffer that is genuinely ambiguous, and the ambiguity is not
academic:

```cpp
std::vector<std::pair<int, int>> v = ...;
kamping::v2::send(v | views::with_type(pair_dt), dst, comm);  // send pairs as a struct
```

`pair<int,int>` also matches `value_destination_pair` (element 1 is a valid
rank, element 0 is not a range). If the payload annotation lived on
`mpi_type()`, a normal `send` (which reads `mpi_type()`) could no longer tell
"send each element as `pair_dt`" from "send each value slot as `int_dt`".

So the payload type needs its **own channel**, orthogonal to `mpi_type()`:

| Adaptor              | Channel set        | Meaning                                   |
|----------------------|--------------------|-------------------------------------------|
| `with_type(dt)`      | `mpi_type()`       | MPI type of *this buffer's elements*      |
| `with_value_type(dt)`| `mpi_value_type()` | MPI type of the *payload / value slot*    |

A normal `send`/`recv` reads `mpi_type()` only, so `with_value_type` never
disturbs it. Structured consumers (`flatten_v`, `request_reply`) read
`mpi_value_type()`.

---

## New buffer-protocol accessor: `value_type`

CHANGEME: this should not become part of the buffer protocol, but should live in the v2 layer, since it is only used by v2 machinery.

Add `mpi::experimental::value_type(buf)` alongside `type(buf)`, with the same
three-tier dispatch as the other accessors:

1. `buffer_traits<T>` specialization
2. `t.mpi_value_type()` member
3. fallback: deduce the builtin MPI type from the **payload element type**
   (the shared resolver below)

`has_mpi_value_type<T>` mirrors `has_mpi_type<T>` and is used by consumers to
decide between an explicit annotation and the deduced fallback.

The `value_type` accessor is *new vocabulary*; it is only meaningful where a
buffer has a payload distinct from its element. It does not replace `type()`
anywhere — normal single-buffer ops continue to read `type()`.

---

## Shared payload resolver

The C++ payload element type is already computed by `flat_element_t` in
`flatten_v_view.hpp` for all flattenable shapes (nested / sparse /
value-destination). It is currently private to that header, and `type_pool.hpp`
has a *separate* resolver, `mpi_element_type_t` (plain `range_value_t` /
`value_type`, no structural interpretation).

**Hoist** `flat_element_t` + the `flattenable_send_buffer` concepts into a
shared header (`kamping/v2/views/payload.hpp`), renamed `payload_element_t` to
avoid colliding with the pre-existing (and, as corrected below, still-needed)
`mpi_element_type_t` name:

```cpp
template <typename R> struct payload_element;             // -> C++ payload type
// nested / sparse / value_destination : as flat_element today
// plain flat buffer                   : range_value_t<R> / R::value_type   (flat fallback tier)
template <typename R> using payload_element_t = typename payload_element<R>::type;
```

`value_type()`'s tier-3 fallback deduces the builtin MPI type from
`payload_element_t<R>`.

**Correction (post-#66, see #67):** the first cut of this design *removed*
`mpi_element_type_t`, redirecting `with_pool`/`with_auto_pool` at
`payload_element_t` — reasoning that `payload_element_t`'s own flat-fallback
tier is a strict superset of what `mpi_element_type_t` did. That reasoning is
wrong: `with_pool`/`with_auto_pool` mean "this buffer's own element type,
period" (the caller has already committed to treating `R` as flat), while
`payload_element_t`'s structural tiers exist specifically to *reinterpret* an
`R` that might still be structured. Those two intents coincide for a genuinely
flat `R`, but diverge silently whenever `R`'s flat element type happens to
*structurally resemble* one of the flattenable shapes by coincidence — e.g. a
plain `std::vector<std::pair<VId, VId>>` (an ordinary key/value buffer)
satisfies `value_destination_pair_buffer` purely because its second field is
an integer, which is all the `rank` concept checks for. `payload_element_t`
then silently resolves to just the pair's first field, and `with_auto_pool`
attaches a datatype half the true element width — no compile error, no MPI
error at low rank counts, just a truncated buffer.

`mpi_element_type_t` is therefore **kept**, unchanged, in `payload.hpp`
(not removed), and remains what `with_pool`/`with_auto_pool` resolve through.
`payload_element_t` is genuinely new vocabulary alongside it — used only by
`with_value_pool`/`with_auto_value_pool`, `flatten_v_view` (on `Source`, never
on itself — see the `already_flat_buffer` note under Consumer wiring below),
and `dstl::request_reply`.

---

## `with_value_type` view

`with_value_type_view<Base>` mirrors `with_type_view` but writes the other
channel: stores an `MPI_Datatype`, exposes `mpi_value_type()`, forwards the
range (`begin`/`end`), and leaves `mpi_type()` to the base (which, for a raw
structured range, has none). Ownership/borrow semantics follow `all()` exactly
as `with_type_view` does.

```cpp
value_dest_pairs | views::with_value_type(int_dt)   // -> mpi_value_type() == int_dt, still a value_destination_pair_buffer
```

Adaptor factory `kamping::v2::views::with_value_type`, analogous to `with_type`.

`view_interface` must forward `mpi_value_type()` from `base()` exactly as it
already forwards `mpi_type()` / `mpi_count()` (see `view_interface.hpp`), so the
annotation survives further wrapping (`... | with_value_type(dt) | resize`, GPU
/ serialization layers, …). Without the forwarder an outer view would shadow the
value-type channel.

### Datatype lifetime

Same contract as `with_type` / `with_pool`: `with_value_type_view` only stores
the `MPI_Datatype` **handle**; the committed type is owned elsewhere (a
`type_pool` or a `ScopedDatatype`) and must outlive the operation. For
**non-blocking** ops the buffer moves into `iresult`, which does **not** take
ownership of the datatype — the caller upholds the lifetime contract. This is
consistent with `iresult` not storing the `MPI_Op` for non-blocking reductions
either; auxiliary MPI objects are the caller's responsibility for now. (A future
owning-type view could bundle a `ScopedDatatype` into the buffer if this proves
error-prone; out of scope here.)

---

## `type_pool`: one-shot registration

`with_value_pool` (and `with_pool`) can only attach a type the pool can resolve.
Today a derived type is only registrable if a global `mpi_type_traits<T>` exists
(`register_type<T>()` requires `has_static_type_v<T>`). That is too restrictive
for the common case "I have an `MPI_Datatype` for `T` and just want the pool to
hold it" — e.g. a byte-blob type for a struct without writing a trait
specialization.

Add a one-shot overload:

```cpp
// register a caller-supplied datatype for T, no mpi_type_traits<T> required.
template <typename T>
MPI_Datatype register_type(MPI_Datatype dt);

// usage:
pool.register_type<MyStruct>(kamping::types::byte_serialized<MyStruct>::data_type());
```

Semantics:

- `T` is used **purely as the lookup key** (`type_index(typeid(T))`); the
  `has_static_type_v<T>` requirement is dropped for this overload.
- The pool wraps `dt` in a `ScopedDatatype`, which **commits** it on
  construction and **frees** it on destruction. The pool owns the handle; the
  caller must not free it. Passing an already-committed handle is fine
  (`MPI_Type_commit` on a committed type is a no-op per the MPI standard); the
  existing path already relies on `ScopedDatatype` to commit the uncommitted
  type returned by `data_type()`.
- Precondition: do **not** pass a predefined/named handle (e.g. `MPI_INT`) —
  `ScopedDatatype` would attempt to free it (cf. the `FIXME` in
  `scoped_datatype.hpp`). One-shot registration is for derived types.

`find<T>()` must become **trait-free** too, otherwise a one-shot type can never
be looked up:

```cpp
template <typename T>
std::optional<MPI_Datatype> find() const {
    if constexpr (has_static_type_v<T> && !mpi_type_traits<T>::has_to_be_committed)
        return mpi_type_traits<T>::data_type();   // builtin shortcut
    else {
        auto it = _types.find(std::type_index(typeid(T)));   // no trait required
        return it == _types.end() ? std::nullopt : std::optional{it->second.data_type()};
    }
}
```

The trait-only `register_type<T>()` overload is unchanged and coexists.

## Pool-backed variants

Mirror `with_pool` / `with_auto_pool`, but resolve the **payload** type via the
shared resolver and set the value-type channel:

```cpp
inline constexpr adaptor<1, with_value_pool_fn>      with_value_pool{};      // requires prior register_type<payload_t>()
inline constexpr adaptor<1, with_auto_value_pool_fn> with_auto_value_pool{}; // registers payload_t lazily
```

Both resolve `payload_element_t<R>`, look it up / register it in the
`type_pool`, and wrap with `with_value_type_view`. The existing
`with_pool` / `with_auto_pool` are unchanged (element channel).

Lifetime caveat is identical to `with_pool`: the committed `MPI_Datatype` is
owned by the pool, the view only borrows the handle, so the pool must outlive
the operation.

---

## Consumer wiring

**`flatten_v_view`** — currently reports the flat buffer's `mpi_type()` via
builtin deduction of `flat_element_t` (`view_interface` forwards from
`base()`). Change: if `Source` provides `mpi_value_type()`, report that as the
flattened buffer's `mpi_type()`; otherwise keep the deduced fallback. A raw
`vector<vector<int>>` has no `mpi_value_type()`, so the check cleanly
discriminates annotated from unannotated sources.

**Correction (post-#66, see #67), two related gaps in the original plan:**

1. `flatten_v_view` itself must **never** satisfy `nested_send_buffer` /
   `sparse_nested_send_buffer` / `value_destination_pair_buffer` again, no
   matter what its own flattened element type looks like — it is already
   resolved, flat data, not a still-structured source. Without this, the same
   `std::pair<VId, VId>` coincidence above makes a `flatten_v_view` of such
   pairs itself satisfy `value_destination_pair_buffer`, which (a) makes
   `payload_element_t<flatten_v_view<...>>` misresolve exactly like the
   `with_pool` case, and (b) lets a `flatten_v_view` wrongly satisfy
   `dstl::request_reply`'s `value_destination_pair_buffer Requests`
   constraint — nonsensical, since a flattened buffer has no per-item
   destination left to give a round-trip primitive. Fixed via an opt-out
   marker, `already_flat_buffer<T>` (mirrors the existing
   `enable_borrowed_buffer<T>` idiom), specialized `true` for `flatten_v_view`
   and checked by the three leaf concepts (not just the `flattenable_send_buffer`
   umbrella, so any future direct consumer of e.g. `value_destination_pair_buffer`
   — `dstl::request_reply` included — is automatically covered).

2. `flatten_v_view` must **not** inherit `view_interface`'s generic
   `mpi_value_type()` forwarder (which reaches into `base()`, i.e. `FlatBuf`).
   That forwarder exists so an annotation survives *further wrapping* layers
   sitting between an annotated source and its consumer (`resize`, GPU/
   serialization adaptors, …) — `flatten_v_view` is not such a layer, it *is*
   the terminal consumer, already folded into its own `mpi_type()` override
   above. Left inherited, `has_mpi_value_type<flatten_v_view<...>>` comes out
   true via the exact same mechanism as (1) applied to `FlatBuf` instead (a
   plain `std::vector<std::pair<VId, VId>>` also satisfies
   `payload_element_is_builtin` in `value_type()`'s tier-3 fallback, since its
   misresolved "payload" is just the first field — an integer). Fixed by
   shadowing the name with a deleted member, `void mpi_value_type() const =
   delete;`, which hides the inherited template member regardless of
   signature.

**`dstl::request_reply`** (new) — consumes the `value_destination_pair` buffer
natively. Per value it sends with
`has_mpi_value_type<Buf> ? value_type(buf) : builtin_type<payload_element_t<Buf>>`.

```cpp
type_pool p;
dstl::request_reply(value_dest_pairs | views::with_auto_value_pool(p), ...);   // native, no packing
dstl::sparse_alltoall(nested | views::with_value_type(dt) | flatten_v(), ...); // packed path
```

### Receive side / round-trip contract

`value_type` is a **send-side** concept (structured send buffers). The receiver
in `request_reply` / `sparse_alltoall` works through `probe_result` + a *flat*
recv buffer, which already carries a single `mpi_type()` — so the existing
`with_type` (or builtin deduction) covers it; no `value_type` on the recv side.

The contract is therefore explicit and the caller's to uphold: the sender's
value type and the receiver's recv-buffer type must match (MPI matches by type
signature). The asymmetry is intentional — `with_value_type(dt)` on the send,
`with_type(dt)` on the recv:

```cpp
// sender:   value payloads carry dt
dstl::request_reply(pairs | views::with_value_type(dt), on_reply, comm);
// receiver: flat recv buffer carries the same dt
on_message = [&](probe_result pr) {
    auto buf = std::vector<...>(pr.count(dt));          // count in units of dt
    mpi::experimental::mrecv(std::move(pr), buf | views::with_type(dt));
};
```

`probe_result::count(MPI_Datatype)` (the explicit-type variant from
`sparse_alltoall.md`) lets the receiver reuse the same handle rather than
re-deducing from a `T`, keeping both ends on one datatype.

---

## Migration / call sites that switch to `value_type`

`mpi::experimental::value_type` is new vocabulary, so very little existing code
"switches" to it — the deliberate scope:

- **`flatten_v_view`** — the one existing consumer that starts *reading*
  `value_type(source)` (to honor a `with_value_type` annotation through the
  pack).
- **`type_pool.hpp`** — `mpi_element_type_t` moves to the shared `payload.hpp`
  alongside `payload_element_t` but is **not** removed and **not** replaced
  (see the Shared payload resolver correction above): the *element*-channel
  pool adaptors (`with_pool`/`with_auto_pool`) keep resolving through it,
  unchanged in meaning; the new *value*-channel variants
  (`with_value_pool`/`with_auto_value_pool`) are added beside them, resolving
  through `payload_element_t` instead.
- **Normal `send`/`recv`/`bcast`/…** — unchanged; they read `type()`, never
  `value_type()`. Switching them would be wrong (it would reintroduce the
  ambiguity above).
- **Variadic ops (`alltoallv`, …)** — unchanged; they receive already-flat
  `data_buffer_v` input, so no payload-vs-element distinction arises.

---

## Resolved questions

- **Channel name**: `value_type()` / `has_mpi_value_type` / `value_type_traits` /
  `with_value_type(_view)` / `with_value_pool` / `with_auto_value_pool` — "value"
  wins for this whole public surface. It reads best at the actual call sites
  (`nested | views::with_value_type(dt) | flatten_v()` reads as a plain sentence;
  `with_payload_type` drags in more jargon for no gain there), and the collision
  with `T::value_type` is a non-issue in practice — different syntax forms
  (function call vs. member-type access) make the two unambiguous on sight.
  `payload_element_t` / `payload_element<R>` / `payload.hpp` keep "payload" —
  different audience (internal C++-type resolver used in view factories and
  concept checks, not something a caller writes in a pipe chain), independently
  settled, no real conflict with the public "value" naming above. Pre-existing
  `with_pool` / `with_auto_pool` (element channel) are unaffected and out of
  scope for this decision.
- **`register_type<T>(dt)` collision policy**: idempotent keep-existing, against
  this doc's original "assert" leaning. Reasoning: a caller must be able to call
  `register_type<T>(dt)` repeatedly for the same `T` (e.g. once per
  `request_reply()` invocation) without erroring — the caller has no cheap way to
  know in advance whether a shared pool has already seen `T`, and a helper like
  `byte_serialized<T>::data_type()` constructs a fresh derived datatype on every
  call. Assert-against-re-register would make that pattern always fail on the
  second call. The redundant handle is freed immediately (`MPI_Type_free`, legal
  on both committed and uncommitted derived types) rather than leaked or
  silently kept idle. The tradeoff this accepts: two call sites registering
  genuinely *different* (incompatible) layouts under the same `T` would have the
  second one silently ignored — but that can only happen if one of them is
  already wrong, since `T`'s actual memory layout is fixed by the compiler; any
  two *correct* descriptions of it are functionally equivalent, so this isn't a
  new failure mode, just the same "MPI won't validate your datatype for you"
  property every `with_type`/`with_pool` call already has.

## Open questions

- Whether `data_buffer_v` packed inputs should also accept `with_value_type`
  (currently no need — they are already flat with a single `mpi_type()`).
