# Receive buffers that size themselves

The views in {doc}`views` derive everything from the buffer they wrap, without communication. Often, however, the
receiving side does not know how much data will arrive: the size of a message is determined by its sender, and in
a variable-size collective each process contributes a different number of elements. With plain MPI, you probe the
message or exchange the counts yourself, then allocate the receive buffer.

KaMPIng-v2 can do this for you, but only where you ask for it. A *deferred* buffer does not know its receive
count(s) yet, and you request one explicitly at the call site:

- `views::resize` defers a single receive count. The container is resized to fit before receiving.
- `views::auto_counts` defers the per-rank counts of a variable-size collective. They are exchanged before the
  operation.

```cpp
std::vector<int> buf;                       // size unknown
v2::recv(buf | views::resize, source, comm);

std::vector<int> all;
v2::allgatherv(mine, all | views::auto_counts | views::auto_displs | views::resize_v,
               comm);
```

`v2::auto_recv_v<int>()` is shorthand for a new `std::vector<int>` with these three views, and
`v2::auto_recv<int>()` for one with `views::resize`.

A buffer passed as an rvalue is handed back by the operation, still wrapped in its views. `underlying()` returns
the container inside:

```cpp
std::vector<int> data = v2::recv(v2::auto_recv<int>(), source, comm).underlying();
auto [_, all] = v2::allgatherv(mine, v2::auto_recv_v<int>(), comm);
for (int x : all.underlying()) { ... }
```

## How the count is obtained

Before calling MPI, every `kamping::v2` operation calls the customization point `infer()` with its buffers and the
communicator. For plain buffers it does nothing. For a deferred buffer it obtains the missing count in the cheapest
way the operation allows:

- **without communication**, where the count is already determined: in `allreduce`, the receive count equals the
  send count;
- **with a probe**, for point-to-point receives: the message is matched with `MPI_Mprobe`, the buffer is resized,
  and the matched message is received with `MPI_Mrecv`;
- **with one count exchange**, for variable-size collectives: e.g. an `allgather` of the send counts before
  `allgatherv`.

KaMPIng-v2 never turns a plain buffer into a deferred one and never communicates more than you asked for. Naming
the view makes the extra communication visible at the call site, while the probe or exchange itself is done for
you. Code that provides all counts itself pays nothing: the deferred-buffer code is compiled away.

## Composing deferred buffers

Deferred buffers are views, so they compose with the others. In
`all | views::auto_counts | views::auto_displs | views::resize_v`, only `auto_counts` is deferred: `infer()` fills
in its counts through the count exchange. `auto_displs` then computes the displacements from them, and `resize_v`
resizes `all` once counts and displacements are known. Nothing is allocated or computed before that.

By default, `auto_counts` and `auto_displs` keep their counts and displacements in an owned `std::vector<int>`.
To reuse your own storage, pass it — `views::auto_counts(counts)`, or `views::auto_counts(resize, counts)` to have
it resized.

## Serialization

Types without an MPI datatype, such as a `std::map`, can be sent with the Cereal adapter: `views::serialize`
serializes an object for sending, and `views::deserialize<T>()` is a deferred receive buffer that deserializes what
arrives. Because it builds on the same buffer and deferred-buffer concepts, it works with every operation. See
{doc}`ecosystem`.

## Extending `infer()`

`infer()` is found by argument-dependent lookup, so you can add overloads for your own buffer types, e.g. to
exchange additional metadata before an operation. See {doc}`../reference/infer`.
