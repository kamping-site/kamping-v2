# Views

The C++20 ranges library introduced *range adaptors*: lazy, composable transformations written with a pipe.
`std::views::take(n)` does not copy or modify anything; it produces a lightweight view of the first `n` elements of
a range. KaMPIng-v2 applies the same idea to MPI buffers. Its views attach or override MPI properties — the
datatype, the element count, per-rank counts and displacements — and compose with each other and with the standard
views:

```cpp
v2::send(data | std::views::take(n), dest, comm);                 // standard view
v2::send(data | views::with_type(my_type), dest, comm);           // override the datatype
v2::send(data | std::views::take(n) | views::with_type(my_type),  // both
         dest, comm);
```

Every view produces a buffer again, so it works with every operation. All views live in `kamping::v2::views`; the
{doc}`reference <../reference/views>` lists them.

## How views work

A view wraps a buffer and forwards all accessors to it, except those it is responsible for:
`views::with_type(dt)` returns `dt` from `type()` and passes `ptr()` and `count()` through unchanged. Views that
compute something do so lazily, when an operation first asks for it, and only once: `views::auto_displs` computes
the displacements as the exclusive prefix sum of the counts. Composing views therefore allocates and computes
nothing, and costs nothing if their result is never used.

## Ownership

Like the standard views, a view borrows an lvalue and owns an rvalue:

```cpp
std::vector<int> v = ...;
auto a = v | views::with_type(dt);                 // refers to v
auto b = std::move(v) | views::with_type(dt);      // owns the vector
```

This carries through to the results of operations: buffers that were moved in are handed back by value, borrowed
buffers by reference (see {doc}`nonblocking`).

## Attaching per-rank counts and displacements

For variable-size collectives, `with_counts` and `with_displs` attach existing counts and displacements, and
`auto_displs` computes the displacements from the counts:

```cpp
// counts and displacements you already have
auto a = data | views::with_counts(counts) | views::with_displs(displs);
// displacements computed from the counts
auto b = data | views::with_counts(counts) | views::auto_displs;
```

If the displacements you provide are monotonically non-decreasing, say so with
`views::with_displs(monotonic, displs)`: a receive buffer can then be sized from the last count and displacement
alone.

`views::flatten_v()` builds a variable-size send buffer from a range of per-rank ranges, and `views::chunked` splits
a variable-size buffer back into per-rank `std::span`s:

```cpp
std::vector<std::vector<int>> per_rank = ...;   // per_rank[i] goes to rank i
auto [_, recv] = v2::alltoallv(per_rank | views::flatten_v(),
                               v2::auto_recv_v<int>(), comm);
for (auto from_rank : recv | views::chunked) { ... }  // std::spans
```

## Single values

`views::ref_single(x)` makes a single value a buffer of one element:

```cpp
int sum = local;
v2::allreduce(v2::inplace, views::ref_single(sum), comm);
```
