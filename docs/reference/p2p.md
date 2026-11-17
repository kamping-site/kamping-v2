# Point-to-point communication

All operations take the communicator last. Arguments with an obvious default can be left out in front of it: the
tag (`0` for sends, any tag for receives) and the source of a receive (any source). See {doc}`../guide/conventions`.

Blocking operations return the buffers they were given: owned buffers (passed as rvalues) by value, borrowed buffers
(lvalues) by reference. Non-blocking operations return an `iresult` that hands the buffers back on completion; see
{doc}`results`.

## Sending

```{doxygenoverloads} kamping::v2::send
```

The send mode is selected with a tag as first argument, e.g. `v2::send(v2::send_mode::sync, buf, dest, comm)`:

```{doxygenvariable} kamping::v2::send_mode::standard
```

```{doxygenvariable} kamping::v2::send_mode::buffered
```

```{doxygenvariable} kamping::v2::send_mode::sync
```

```{doxygenvariable} kamping::v2::send_mode::ready
```

```{doxygenvariable} kamping::v2::DEFAULT_SEND_TAG
```

## Receiving

```{doxygenoverloads} kamping::v2::recv
```

## Sending and receiving at once

```{doxygenoverloads} kamping::v2::sendrecv
```

With one buffer for both directions:

```{doxygenoverloads} kamping::v2::sendrecv_replace
```

## Special ranks and tags

The wildcards and the null process have types of their own, so a send to `any_source` or with `any_tag` does not
compile. The raw `MPI_ANY_SOURCE`, `MPI_ANY_TAG` and `MPI_PROC_NULL` are still accepted. A receive from `proc_null`
transfers nothing, which its status reports with `from_proc_null()`.

```cpp
v2::recv(buf | views::resize, v2::any_source, v2::any_tag, comm);
v2::send(buf, v2::proc_null, comm);       // completes immediately
v2::send(buf, v2::any_source, comm);      // compile error
```

```{doxygenvariable} kamping::v2::any_source
```

```{doxygenvariable} kamping::v2::any_tag
```

```{doxygenvariable} kamping::v2::proc_null
```

The wrappers state the role of each rank and tag with these concepts:

```{doxygenconcept} kamping::v2::sender
```

```{doxygenconcept} kamping::v2::destination
```

```{doxygenconcept} kamping::v2::message_tag
```

## Neighbor ranks

`periodic_rank` and `nonperiodic_rank` hold a rank together with the size of its communicator and can be shifted by
an offset. They can be passed wherever a rank is expected. A `periodic_rank` wraps around at the ends, as in a ring;
a `nonperiodic_rank` beyond the ends stands for `MPI_PROC_NULL`, so communication with a missing neighbor is skipped:

```cpp
auto const me = comm.periodic_rank();
v2::sendrecv_replace(block, me - 1, me + 1, comm);  // shift the blocks one rank to the left
```

Every communicator type has the members `periodic_rank()` and `nonperiodic_rank()` for the rank of the calling
process, and `periodic_rank(r)` / `nonperiodic_rank(r)` for the rank `r`. The constructors do the same and also take
a raw `MPI_Comm` or any other communicator wrapper: `v2::periodic_rank(comm)`, `v2::nonperiodic_rank(r, comm)`.

```{doxygenclass} kamping::v2::periodic_rank
:members:
```

```{doxygenclass} kamping::v2::nonperiodic_rank
:members:
```

## Probing

```{doxygenoverloads} kamping::v2::probe
```

```{doxygenoverloads} kamping::v2::iprobe
```

```{doxygenclass} kamping::v2::probe_result
:members:
```

### Matched probe and receive

```{doxygenoverloads} kamping::v2::mprobe
```

```{doxygenoverloads} kamping::v2::improbe
```

```{doxygenoverloads} kamping::v2::mrecv
```

```{doxygenoverloads} kamping::v2::imrecv
```

## Non-blocking operations

```{doxygenoverloads} kamping::v2::isend
```

```{doxygenoverloads} kamping::v2::irecv
```

```{doxygenoverloads} kamping::v2::isendrecv
```
