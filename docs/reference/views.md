# Views

All views live in `kamping::v2::views` and compose with `|`. Each one can be called with the buffer as the first
argument or applied with `|`, like a `std::ranges` adaptor: `views::with_counts(buf, counts)` is the same as
`buf | views::with_counts(counts)`. Views without arguments, such as `views::resize`, are applied without
parentheses. Buffers passed as lvalues are borrowed, rvalues are moved into the view.

Views are lazy: composing them allocates and computes nothing, and their metadata is computed on first access.

## Getting the container back

Operations return the buffers they were given, so a buffer passed through views comes back as a view. Every view
has {cpp:func}`underlying() <kamping::v2::view_interface::underlying>`, which returns the container at the bottom of
the chain: moved out of an rvalue, by reference from an lvalue.

```cpp
std::vector<int> data = v2::recv(std::vector<int>{} | views::resize, source, comm).underlying();

auto [_, recv] = v2::allgatherv(local, v2::auto_recv_v<int>(), comm);
std::vector<int>& all = recv.underlying();
```

## Attaching metadata

```{doxygenview} kamping::v2::views::with_type
```

```{doxygenview} kamping::v2::views::with_count
```

:::{doxygenview} kamping::v2::views::with_counts
Example:

```cpp
std::vector<int> data = ..., counts = ...;
auto send = data | views::with_counts(counts) | views::auto_displs;
v2::alltoallv(send, v2::auto_recv_v<int>(), comm);
```
:::

```{doxygenview} kamping::v2::views::with_displs
```

```{doxygenview} kamping::v2::views::auto_displs
```

```{doxygenview} kamping::v2::views::with_value_type
```

## Receive buffers that size themselves

```{doxygenview} kamping::v2::views::resize
```

```{doxygenview} kamping::v2::views::auto_counts
```

```{doxygenview} kamping::v2::views::resize_v
```

```{doxygenview} kamping::v2::views::auto_recv_v
```

```{doxygenoverloads} kamping::v2::auto_recv_v
```

```{doxygenoverloads} kamping::v2::auto_recv
```

## Datatypes from a type pool

See {doc}`types`.

```{doxygenview} kamping::v2::views::with_pool
```

```{doxygenview} kamping::v2::views::with_auto_pool
```

```{doxygenview} kamping::v2::views::with_value_pool
```

```{doxygenview} kamping::v2::views::with_auto_value_pool
```

## Building and splitting variable-size buffers

```{doxygenoverloads} kamping::v2::views::flatten_v
```

```{doxygenview} kamping::v2::views::chunked
```

## Single values

```{doxygenoverloads} kamping::v2::views::ref_single
```

## Tags

```{doxygenvariable} kamping::v2::resize
```

```{doxygenvariable} kamping::v2::monotonic
```
