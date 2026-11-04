# Collective operations

Collectives follow the same conventions as point-to-point operations: the communicator comes last, the root
defaults to rank 0 and the reduction operation to `std::plus<>`. Rooted collectives also have a shorthand for
non-root ranks that leaves out the buffer only the root uses, e.g. `gather(sbuf, comm)` on a non-root rank. See
{doc}`../guide/conventions`.

Variable-size collectives (`gatherv`, `scatterv`, `allgatherv`, `alltoallv`) take buffers with per-rank counts and
displacements. If a receive buffer defers its counts (`views::auto_counts`, `auto_recv_v<T>()`), they are exchanged
before the operation; see {doc}`../guide/receive-buffers`.

Operations with a send and a receive buffer return a {cpp:class}`kamping::v2::result`, which can be unpacked with a
structured binding: `auto [sbuf, rbuf] = v2::allgather(...)`.

## Synchronization

```{doxygenoverloads} kamping::v2::barrier
```

```{doxygenoverloads} kamping::v2::ibarrier
```

## Broadcast

```{doxygenoverloads} kamping::v2::bcast
```

## Gather and scatter

```{doxygenoverloads} kamping::v2::gather
```

```{doxygenoverloads} kamping::v2::gatherv
```

```{doxygenoverloads} kamping::v2::scatter
```

```{doxygenoverloads} kamping::v2::scatterv
```

```{doxygenoverloads} kamping::v2::allgather
```

```{doxygenoverloads} kamping::v2::allgatherv
```

## All-to-all

```{doxygenoverloads} kamping::v2::alltoall
```

```{doxygenoverloads} kamping::v2::alltoallv
```

## Reductions

```{doxygenoverloads} kamping::v2::reduce
```

```{doxygenoverloads} kamping::v2::allreduce
```

```{doxygenoverloads} kamping::v2::scan
```

```{doxygenoverloads} kamping::v2::exscan
```

### Reduction operations

An operation is an `MPI_Op`, a standard function object such as `std::plus<>` or `std::multiplies<>` (mapped to
the builtin operation where one exists), or any callable wrapped with `make_op`.

```{doxygenconcept} kamping::v2::valid_op
```

```{doxygenoverloads} kamping::v2::make_op
```

```{doxygenvariable} kamping::v2::commutative
```

```{doxygenvariable} kamping::v2::non_commutative
```

## Special buffers

```{doxygenvariable} kamping::v2::inplace
```

```{doxygenvariable} kamping::v2::null_buf
```

```{doxygenvariable} kamping::v2::null_buf_v
```

```{doxygenoverloads} kamping::v2::auto_null_recv_v
```

```{doxygenvariable} kamping::v2::bottom
```
