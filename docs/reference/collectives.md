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

An operation is an `MPI_Op`, a function object of a predefined operation such as `std::plus<>` or `ops::max<>`
(mapped to `MPI_SUM`, `MPI_MAX`, ...), or an operation object, which `make_op` creates from a stateless functor or
an MPI callback. Operation objects are also callable on single values, so the same object can
combine values locally.

```{doxygenconcept} kamping::v2::valid_op
```

```{doxygenconcept} kamping::v2::builtin_op
```

```{doxygenconcept} kamping::v2::reduction_op
```

```{doxygenoverloads} kamping::v2::as_local_op
```

```{doxygenoverloads} kamping::v2::builtin_op_handle
```

```{doxygenstruct} kamping::v2::ops::max
```

```{doxygenstruct} kamping::v2::ops::min
```

```{doxygenstruct} kamping::v2::ops::logical_xor
```

```{doxygenoverloads} kamping::v2::make_op
```

```{doxygenclass} kamping::v2::functor_op
:members:
```

```{doxygenconcept} kamping::v2::reduction_functor
```

```{doxygentypedef} kamping::v2::user_function
```

```{doxygenclass} kamping::v2::op
:members:
```

```{doxygenclass} kamping::v2::op_view
:members:
```

```{doxygenclass} kamping::v2::op_accessors
:members:
```

```{doxygenoverloads} kamping::v2::reduce_local
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
