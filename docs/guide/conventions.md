# Calling conventions

All operations in `kamping::v2` follow the same argument order: the communicator comes last and is always
required, followed only by an optional `MPI_Status*`. Arguments with an obvious default can be left out in front of
it, by passing *fewer* arguments:

```cpp
v2::send(buf, dest, comm);           v2::send(buf, dest, tag, comm);
v2::recv(buf, comm);                 v2::recv(buf, source, comm);
v2::recv(buf, source, tag, comm);
v2::bcast(buf, comm);                v2::bcast(buf, root, comm);
v2::reduce(s, r, comm);              v2::reduce(s, r, op, comm);
v2::reduce(s, r, op, root, comm);
```

A left-out argument takes the obvious value:

| Argument | Default |
|----------|---------|
| send tag | `DEFAULT_SEND_TAG` (0) |
| source of a receive | `MPI_ANY_SOURCE` |
| tag of a receive | `MPI_ANY_TAG` |
| root | 0 |
| reduction operation | `std::plus<>` |

Rooted collectives also have a shorthand for non-root processes, without the buffer that only the root uses:
`gather(sbuf, [root,] comm)`, `scatter(rbuf, [root,] comm)`, `scatterv(rbuf, [root,] comm)` and
`reduce(sbuf, [op, [root,]] comm)`. `gatherv` has none, since what a non-root process passes depends on whether
the root's counts are deferred.

A request you manage yourself is passed as the *first* argument of a non-blocking operation:
`v2::isend(request, buf, dest, comm)`. The send mode, if not standard, is also passed first:
`v2::send(v2::send_mode::sync, buf, dest, comm)`.

## Why not default arguments?

On some MPI implementations, `MPI_Comm`, `MPI_Request`, `MPI_Op` and the other handles are plain `int`s, just like
ranks and tags. With default arguments in the middle of the list, a communicator could silently be taken for a tag.
With the rules above, the forms of an operation differ in their number of arguments or in an argument that is never
an `int`, so every call is unambiguous. For the same reason there is no `reduce(s, r, root, comm)`: a root cannot be
told apart from an `MPI_Op`.

## Return values

Operations return the buffers they were given (see {doc}`nonblocking`): a single buffer directly, a send and a
receive buffer as a `result` that can be unpacked with a structured binding, and non-blocking operations an
`iresult`.
