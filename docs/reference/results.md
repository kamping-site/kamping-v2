# Results and non-blocking operations

Buffers passed as rvalues are moved into an operation and handed back by value; buffers passed as lvalues are
borrowed and handed back by reference. Blocking operations with two buffers return them in a `result`;
non-blocking operations return an `iresult`, which owns the request and keeps owned buffers out of reach until the
operation has completed. See {doc}`../guide/nonblocking`.

## Blocking results

```{doxygenstruct} kamping::v2::result
:members:
```

## Non-blocking results

```{doxygenclass} kamping::v2::iresult< Buf >
:members:
```

```{doxygenclass} kamping::v2::iresult< SBuf, RBuf >
:members:
```

## Requests

```{doxygenclass} kamping::v2::request
:members:
```

```{doxygenclass} kamping::v2::request_view
:members:
```

```{doxygenclass} kamping::v2::request_vector
:members:
```

## Completion

The functions completing several requests take any writable contiguous range of `MPI_Request`s, such as a
`request_vector` or a `std::vector<MPI_Request>`, and of `MPI_Status`es for the statuses:

```{doxygenconcept} mpi::experimental::request_range
```

```{doxygenconcept} mpi::experimental::output_array_of
```

```{doxygenconcept} mpi::experimental::output_array_or_ignore_of
```

```{doxygenoverloads} kamping::v2::wait
```

```{doxygenoverloads} kamping::v2::test
```

```{doxygenoverloads} kamping::v2::waitall
```

```{doxygenoverloads} kamping::v2::testall
```

```{doxygenoverloads} kamping::v2::waitany
```

```{doxygenoverloads} kamping::v2::testany
```

```{doxygenclass} kamping::v2::testany_result
:members:
```

```{doxygenoverloads} kamping::v2::waitsome
```

```{doxygenoverloads} kamping::v2::testsome
```

```{doxygenvariable} kamping::v2::statuses_ignore
```
