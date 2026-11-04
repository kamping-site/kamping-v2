# KaMPIng-v2

*Modern C++ MPI bindings: pass your containers, compose what you need.*

Calling MPI from C++ means passing raw pointers, element counts, datatypes and displacement arrays by hand, and
getting one of them wrong usually crashes or hangs. **KaMPIng-v2** lets you pass the objects you already have, such
as a `std::vector`, a `std::span` or a Kokkos view, and works out the pointer, count and datatype for you. When you
need more, you add it with a `|` pipe in the style of `std::ranges` views: a receive buffer that resizes itself,
per-rank counts that are exchanged for you, or a custom datatype.

```cpp
v2::send(data, dest, comm);
v2::recv(buf | views::resize, source, comm);
auto [_, everything] = v2::allgatherv(mine, v2::auto_recv_v<int>(), comm);
```

KaMPIng-v2 is header-only and requires C++20. New here? Start with {doc}`guide/getting-started`.

## How it is built

KaMPIng-v2 is organized in layers on top of the MPI standard:

- **mpi-core** (`mpi::experimental`) is a minimal C++ interface that expresses MPI's own abstractions — data
  buffers, MPI objects and their handles — as C++20 concepts. It mirrors the MPI standard closely: one MPI call per
  function, MPI semantics unchanged, no hidden allocation or communication. It is designed as a reference for
  standardized C++ bindings, and it is the extension point for everything above it.
- **KaMPIng-v2** (`kamping::v2`) builds the convenience on top: composable views that attach MPI metadata to any
  buffer, receive buffers that size themselves, memory-safe non-blocking operations, and defaults for omitted
  arguments. It re-exports the names you need from mpi-core, so you rarely have to name `mpi::experimental`.
- **Ecosystem adapters** make Kokkos views, Thrust and SYCL buffers and Cereal-serialized objects buffers, usable
  with every operation.

This documentation covers the `kamping::v2` interface. Throughout, it follows the principle of *you only pay for
what you use*: convenience that costs something, such as a probe or a count exchange, only happens where you ask
for it at the call site.

KaMPIng-v2 also comes with **dstl**, a library of distributed algorithms and communication patterns built on it,
such as sparse all-to-all exchanges and redistribution.

```{toctree}
:caption: Guide
:maxdepth: 2

guide/getting-started
guide/buffers
guide/views
guide/receive-buffers
guide/nonblocking
guide/objects
guide/datatypes
guide/conventions
guide/ecosystem
```

```{toctree}
:caption: Reference
:maxdepth: 2

reference/p2p
reference/collectives
reference/results
reference/views
reference/buffers
reference/environment
reference/types
reference/ecosystem
reference/infer
reference/customization
```

```{toctree}
:caption: dstl
:maxdepth: 2

dstl/index
dstl/sparse-alltoall
dstl/alltoallv
dstl/redistribute
dstl/request-reply
dstl/algorithms
```

```{toctree}
:caption: About
:maxdepth: 1

citing
design/design-guide
attribute_caching
```
