# All-to-all-v

## Grid all-to-all-v

```{doxygenoverloads} dstl::grid_alltoallv
```

```{doxygenclass} dstl::grid_comm
:members:
```

```{doxygenconcept} dstl::grid_send_buffer
```

```{doxygenconcept} dstl::grid_recv_buffer
```

### Factoring the grid

```{doxygenconcept} dstl::factoring
```

```{doxygenstruct} dstl::dims_create
:members:
```

```{doxygenstruct} dstl::explicit_dims
:members:
```

```{doxygenstruct} dstl::topology_aware
:members:
```

### Execution policies

```{doxygenstruct} dstl::execution_policy::seq
```

```{doxygenstruct} dstl::execution_policy::par
```

```{doxygenstruct} dstl::execution_policy::par_comm
```

### Receive layouts

```{doxygenstruct} dstl::layout::unordered
```

```{doxygenstruct} dstl::layout::ordered_by_source
```

## Multithreaded flat all-to-all-v

```{doxygenoverloads} dstl::alltoallv
```

```{doxygenclass} dstl::thread_multiple_comm
:members:
```

```{doxygenoverloads} dstl::flat_max_threads
```

## Inspecting the setup

```{doxygenoverloads} dstl::as_string
```

```{doxygenoverloads} dstl::gather_thread_counts
```

```{doxygenoverloads} dstl::local_thread_count
```
