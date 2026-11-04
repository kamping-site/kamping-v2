# Inference and deferred buffers

Before each operation, a `kamping::v2` wrapper calls `infer()` with the operation's tag, its buffers and the
communicator. For plain buffers it does nothing. For *deferred* buffers, which do not know their receive count(s)
yet, it obtains them: by probing the message, by a count exchange, or locally where the count is already determined.
See {doc}`../guide/receive-buffers`.

## Deferred buffers

```{doxygenconcept} kamping::v2::deferred_recv_buf
```

```{doxygenconcept} kamping::v2::deferred_recv_buf_v
```

```{doxygenconcept} kamping::v2::deferred_send_buf_v
```

```{doxygenconcept} kamping::v2::has_monotonic_displs
```

```{doxygenoverloads} kamping::v2::materialize
```

```{doxygenvariable} kamping::v2::supports_matched_probe
```

## The `infer()` customization point

Add an overload of `infer()` for your own buffer types in their namespace; it is found by argument-dependent
lookup. The overloads below are the defaults.

```{doxygenoverloads} kamping::v2::infer
```

## Operation tags

Each operation passes an empty tag type from `kamping::v2::comm_op` as first argument, e.g.
`comm_op::allgatherv{}`, so that `infer()` overloads can be written per operation: `recv`, `mrecv`, `sendrecv`,
`bcast`, `gather`, `gatherv`, `scatter`, `scatterv`, `allgather`, `allgatherv`, `alltoall`, `alltoallv`, `reduce`,
`allreduce`, `scan`, `exscan`.
