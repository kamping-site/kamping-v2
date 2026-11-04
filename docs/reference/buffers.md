# Buffer concepts

A type is a *data buffer* if a pointer, an element count and an MPI datatype can be obtained for it, through the
accessors `ptr()`, `count()` and `type()` (see {doc}`customization`). Variable-size operations such as `alltoallv`
additionally need per-rank counts and displacements, through `counts()` and `displs()`. See {doc}`../guide/buffers`.

Counts and displacements are contiguous ranges of `int`:

```{doxygenconcept} kamping::v2::count_range
```

## Single buffers

```{doxygenconcept} kamping::v2::data_buffer
```

```{doxygenconcept} kamping::v2::send_buffer
```

```{doxygenconcept} kamping::v2::recv_buffer
```

```{doxygenconcept} kamping::v2::send_recv_buffer
```

## Variable-size buffers

```{doxygenconcept} kamping::v2::data_buffer_v
```

```{doxygenconcept} kamping::v2::send_buffer_v
```

```{doxygenconcept} kamping::v2::recv_buffer_v
```

## Spans

Like `std::span`, but for any memory and datatype, and with counts and displacements for the variable-size form.

```{doxygenstruct} kamping::v2::mpi_span
```

```{doxygenstruct} kamping::v2::mpi_span_v
```

```{doxygenstruct} kamping::v2::mpi_cspan
```

```{doxygenstruct} kamping::v2::mpi_cspan_v
```

## Structured send buffers

A *structured* send buffer describes per-destination data instead of a flat sequence: a range of per-rank ranges, of
`(rank, message)` pairs, or of `(value, rank)` pairs. `views::flatten_v` and the dstl algorithms consume them. The
values they send are their *payload*, whose datatype is `value_type()` (attach one with `views::with_value_type`).

```{doxygenconcept} kamping::v2::nested_send_buffer
```

```{doxygenconcept} kamping::v2::sparse_nested_send_buffer
```

```{doxygenconcept} kamping::v2::value_destination_pair_buffer
```

```{doxygenconcept} kamping::v2::buffer_message
```

```{doxygenoverloads} kamping::v2::value_type
```

```{doxygenstruct} kamping::v2::value_type_traits
```
