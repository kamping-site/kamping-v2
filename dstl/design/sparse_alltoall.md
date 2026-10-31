# dstl sparse alltoall: open points

`dstl::sparse_alltoall` (NBX, PCX and RSX protocols, receiving via a callback or into one buffer) is implemented in
`dstl/include/dstl/sparse_alltoall.hpp`, which documents its design. This file keeps the points that are still open,
and one decision that is easy to revisit by accident.

## Staging owning messages

Messages produced by value that own their data (e.g. `views::serialize` via `std::views::transform`)
are currently rejected. The shared send phase could instead move each one into an internal vector
that lives until all sends complete; for inputs satisfying `messages_outlive_iteration` this would
compile away. Until then, materialize first:

```cpp
auto staged = sends | std::views::transform(/* ... serialize ... */) | std::ranges::to<std::vector>();
dstl::sparse_alltoall(staged, callback, tag, comm);
```

## Isolation inside the library

Instead of the tag contract, cache a private duplicate of each user communicator
(`cache<attribute::comm, comm>`) and communicate on it, like MPI collectives do internally. Messages
of different calls would still need separating (alternating internal tags, or a sequence number per
cached communicator), but no longer user traffic.

## Convenience variants

Built on Layer 1, collecting all received messages (e.g. as `(source, data)` pairs, or flat data plus
per-element sources), accepting the algorithm tag like it. Layer 2 covers flat data without sources.

## Decided: no `ordered_by_source`


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
