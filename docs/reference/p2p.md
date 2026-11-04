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
