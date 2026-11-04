# Distributed algorithms

Each algorithm operates on a range that is distributed over the processes of a communicator: every process holds
a contiguous part, and the global range is their concatenation in rank order.

## Reductions

```{doxygenoverloads} dstl::reduce
```

```{doxygenoverloads} dstl::reduce_to_root
```

```{doxygenoverloads} dstl::min
```

```{doxygenoverloads} dstl::max
```

```{doxygenoverloads} dstl::all_of
```

```{doxygenoverloads} dstl::is_sorted
```

## Prefix sums

```{doxygenoverloads} dstl::inclusive_scan
```

```{doxygenoverloads} dstl::exclusive_scan
```

## Shifting

```{doxygenoverloads} dstl::shift_left
```

```{doxygenoverloads} dstl::shift_right
```
