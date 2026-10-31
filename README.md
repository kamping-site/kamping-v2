<p align="center">
  <img src="docs/icon.svg" alt="KaMPIng logo" width="140">
</p>

<h1 align="center">KaMPIng v2</h1>

<p align="center">
  <em>Modern C++ MPI bindings: pass your containers, compose what you need.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/kamping-site/kamping-v2" alt="License">
</p>

---

Calling MPI from C++ means passing raw pointers, element counts, datatypes and displacement arrays by hand, and
getting one of them wrong usually crashes or hangs. **KaMPIng v2** lets you pass the objects you already have,
such as a `std::vector`, a `std::span` or a Kokkos view, and works out the pointer, count and datatype for you.
When you need more, you add it with a `|` pipe in the style of `std::ranges` views: a receive buffer that resizes
itself, per-rank counts that are exchanged for you, or a custom datatype.

```cpp
#include <kamping/v2.hpp>

namespace v2    = kamping::v2;
namespace views = kamping::v2::views;

// Point-to-point: the receiver does not need to know the message size in advance.
v2::send(data, dest, comm);
v2::recv(buf | views::resize, source, comm);

// Variable-size collectives: per-rank counts are exchanged and the receive buffer is sized for you.
auto [_, everything] = v2::allgatherv(mine, v2::auto_recv_v<int>(), comm);
v2::alltoallv(sdata | views::with_counts(scounts) | views::auto_displs(), v2::auto_recv_v<double>(), comm);

// GPU and performance-portability containers are buffers, too.
v2::send(kokkos_view | views::kokkos, dest, comm);
```

KaMPIng v2 is header-only and requires C++20. It is the successor of [KaMPIng](https://github.com/kamping-site/kamping)
and is described in our [EuroMPI 2026 paper](#publications) ([preprint](https://arxiv.org/abs/2606.09102)).

## Highlights

- **Your containers are MPI buffers.** Contiguous ranges of builtin types work as they are, including standard
  views like `std::views::take(n)`. Other types plug in through a trait or a few member functions; no inheritance,
  no wrapper classes.
- **Composable views.** `with_type`, `with_counts`, `auto_displs`, `resize`, `flatten_v`, … attach exactly the MPI
  metadata you need, compose with each other and with `std::views`, and compute lazily.
- **You only pay for what you use.** Convenience is opt-in at the call site: a receive buffer only probes or
  exchanges counts if you pipe it through `views::resize` or ask for `auto_recv_v`. KaMPIng v2 never performs
  communication you did not ask for.
- **Memory-safe non-blocking operations.** `isend`/`irecv` return an `iresult` that keeps an owned buffer out of
  reach until the operation has completed, and catches requests that are never completed.
- **Ready for the HPC ecosystem.** Adapters for Kokkos, Thrust, SYCL and Cereal serialization, built on the same
  buffer protocol as everything else.
- **A standardizable core underneath.** Everything is built on [mpi-core](#mpi-core-the-foundation), a thin
  C++ MPI interface you can also use on its own.

## mpi-core: the foundation

KaMPIng v2 consists of two layers, plus the ecosystem adapters:

| Layer | Namespace | Role |
|-------|-----------|------|
| **mpi-core** (`mpi-core/`) | `mpi::experimental` | The C++ core interface: MPI's own abstractions (data buffers, object handles, procedures) as C++20 concepts. One MPI call per function, MPI semantics unchanged, no hidden allocation or communication. |
| **KaMPIng v2** (`kamping-v2/`) | `kamping::v2` | Convenience and safety on top: buffer views, receive buffers that size themselves, ownership-aware results and `iresult`, and defaults for omitted arguments. |
| **Ecosystem adapters** (`ecosystem/`) | `kamping::v2::views` | Kokkos, Thrust, SYCL and Cereal, as buffers usable with both layers. |

mpi-core is deliberately minimal. It mirrors the MPI standard closely, so that it can serve as a reference for
standardized C++ bindings, and it is the extension point for everything above it: a type that works with mpi-core
works with KaMPIng v2 and with all views. Use it directly when you want exact MPI semantics, are building your own
abstractions, or adopt C++ bindings incrementally in an existing MPI code base. It also accepts native MPI handles
and your existing wrapper types.

```cpp
std::vector<double> v = ...;
mpi::experimental::send(v, dest, tag, MPI_COMM_WORLD);                         // mpi-core: explicit, one MPI call
kamping::v2::recv(buf | kamping::v2::views::resize, source, comm);             // KaMPIng v2: probes, then resizes
```

### Buffers and customization

A type is a data buffer if `ptr()`, `count()` and `type()` are defined for it; variable-size operations such as
`alltoallv` additionally need per-rank `counts()` and `displs()`. Each accessor is looked up in three steps:

1. a `buffer_traits<T>` specialization, for types you don't own;
2. a member function (`mpi_ptr()`, `mpi_count()`, `mpi_type()`, …), for types you control;
3. the standard range interface: `std::ranges::data()`, `std::ranges::size()`, and the MPI datatype of the element
   type.

The steps apply per accessor, so a specialization only provides what cannot be derived. A `thrust::device_vector`
only needs its pointer:

```cpp
template <typename T, typename Alloc>
struct mpi::experimental::buffer_traits<thrust::device_vector<T, Alloc>> {
    static T const* ptr(thrust::device_vector<T, Alloc> const& v) { return thrust::raw_pointer_cast(v.data()); }
    static T*       ptr(thrust::device_vector<T, Alloc>& v)       { return thrust::raw_pointer_cast(v.data()); }
};
```

MPI objects come in non-owning (`comm_view`) and owning, move-only RAII (`comm`) forms, and every argument that
stands for an MPI object also accepts the native handle or your own wrapper type (via `handle_traits`). With
`MPI_ERRORS_RETURN`, MPI errors are thrown as `mpi_error`.

## Views

All views compose with `|` and live in `kamping::v2::views`.

| View | What it does |
|------|--------------|
| `resize` | Resizes the container to fit the incoming data before receiving |
| `with_type(dt)` | Uses the MPI datatype `dt` |
| `with_count(n)` | Uses `n` as the element count |
| `with_counts(counts)` | Attaches per-rank counts (variable-size operations) |
| `with_displs([monotonic,] displs)` | Attaches per-rank displacements |
| `auto_counts()` | Per-rank counts that the operation exchanges for you |
| `auto_displs()` | Computes displacements from the counts (exclusive prefix sum) |
| `resize_v` | Resizes a variable-size receive buffer to fit its counts and displacements |
| `auto_recv_v` | Shorthand for `auto_counts() \| auto_displs() \| resize_v` |
| `flatten_v()` | Turns a range of per-rank ranges (or `(value, rank)` pairs) into one variable-size send buffer |
| `chunked` | The inverse: splits a variable-size buffer into per-rank `std::span`s |
| `ref_single(x)` | Treats a single value as a one-element buffer |
| `with_pool(pool)` / `with_auto_pool(pool)` | Attaches a derived datatype from a `type_pool` |

The factories `auto_recv<T>()` and `auto_recv_v<T>()` create a fresh, owned receive buffer that sizes itself.

Derived datatypes are managed by a `type_pool`, which commits a datatype on first use and frees it with the pool.
For plain structs, `struct_type` derives the layout from the fields via Boost.PFR:

```cpp
struct Point { int id; double x; };
template <>
struct kamping::types::mpi_type_traits<Point> : kamping::types::struct_type<Point> {};

kamping::v2::type_pool pool;
kamping::v2::send(points | views::with_auto_pool(pool), dest, comm);
```

## Receive buffers that size themselves

Before each operation, KaMPIng v2 calls the `infer()` customization point, which exchanges metadata between the
send and receive sides where you asked for it. `views::resize` defers a single receive count: point-to-point
receives probe the message, and collectives derive the count locally where it is already known (e.g.
`allreduce`). `views::auto_counts()` defers the per-rank counts of a collective, which are then obtained with one
count exchange. For plain buffers `infer()` does nothing, and composing views allocates nothing; work happens on
first access. You can overload `infer()` for your own buffer types.

## Non-blocking operations

Buffers passed as rvalues are moved into the operation and handed back; lvalues are borrowed. A non-blocking
operation returns an `iresult` that owns the request and, for rvalues, the buffer, so the buffer can't be touched
while MPI may still use it:

```cpp
auto pending = kamping::v2::irecv(std::vector<int>{} | views::resize, source, comm);
// ... do other work ...
std::vector<int> data = pending.wait().underlying(); // the owned buffer is handed back
```

An `iresult` must be completed with `wait()` or a successful `test()`. Destroying one that is still active prints
a diagnostic and calls `std::terminate()` instead of freeing a buffer MPI may still be writing to.

## Calling conventions

In KaMPIng v2 the communicator always comes last, and arguments with an obvious default can be left out in front
of it:

```cpp
kamping::v2::send(buf, dest, comm);                // tag 0
kamping::v2::recv(buf, comm);                      // any source, any tag
kamping::v2::bcast(buf, comm);                     // root 0
kamping::v2::reduce(s, r, comm);                   // std::plus<>, root 0
kamping::v2::reduce(s, r, MPI_MAX, root, comm);
```

mpi-core mirrors MPI and has no such defaults.

## Ecosystem adapters

| Adapter | Header | Usage |
|---------|--------|-------|
| Kokkos | `<kamping/v2/ecosystem/kokkos.hpp>` | `kv \| views::kokkos`; packs non-contiguous views, composes with `views::resize` |
| Thrust | `<kamping/v2/ecosystem/thrust_buffer_traits.hpp>` | makes `thrust::device_vector` a buffer directly (or use `d_vec \| views::thrust`) |
| SYCL | `<kamping/v2/ecosystem/sycl.hpp>` | inside a host task: `acc \| views::sycl(ih)` |
| Cereal | `<kamping/v2/ecosystem/cereal.hpp>` | `views::serialize` / `views::deserialize<T>()` for types without an MPI datatype |

Device memory from `cudaMalloc` or `sycl::malloc_device` needs no adapter: wrap it in a `std::span`.

## Supported operations

**Point-to-point:** `send` (standard, buffered, synchronous, and ready modes), `recv`, `sendrecv`, `isend`, `irecv`,
`isendrecv`, `probe`, `iprobe`, `mprobe`, `improbe`, `mrecv`, `imrecv`

**Collectives:** `allgather`, `allgatherv`, `allreduce`, `alltoall`, `alltoallv`, `barrier`, `ibarrier`, `bcast`,
`exscan`, `gather`, `gatherv`, `reduce`, `scan`, `scatter`, `scatterv`

**Completion:** `wait`, `waitall`, `waitany`, `waitsome`, and their `test` counterparts

**Objects:** communicators, groups, windows, info objects, sessions (MPI 4), attribute caching, and custom
reduction operations via `make_op`

Tested with MPICH, Open MPI and Intel MPI, using GCC and Clang.

## Getting started

KaMPIng v2 needs CMake 3.25+, a C++20 compiler, and an MPI installation. Include `<kamping/v2.hpp>` for
everything, or individual headers such as `<kamping/v2/collectives/alltoallv.hpp>` to keep compile times down.

### With FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    kamping_v2
    GIT_REPOSITORY https://github.com/kamping-site/kamping-v2.git
    GIT_TAG main
)
FetchContent_MakeAvailable(kamping_v2)

target_link_libraries(my_app PRIVATE kamping::v2)
```

To use only the core layer, add `SOURCE_SUBDIR mpi-core` to `FetchContent_Declare` and link `mpi::core`. Set
`KAMPING_V2_ENABLE_<ADAPTER>` before `FetchContent_MakeAvailable` to get an ecosystem adapter.

### As an installed package

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKAMPING_V2_ENABLE_CEREAL=ON   # enable the adapters you need
cmake --install build --prefix /path/to/prefix
```

```cmake
find_package(kamping-v2 0.1 REQUIRED COMPONENTS v2 cereal)
target_link_libraries(my_app PRIVATE kamping::v2 kamping::v2::cereal)
```

| Component | Target | Contents |
|-----------|--------|----------|
| `core` | `mpi::core` | The core layer |
| `v2` | `kamping::v2` | The ergonomic layer (loads `core`) |
| `dstl` | `dstl::dstl` | Distributed algorithms (loads `v2`) |
| `cereal`, `kokkos`, `thrust`, `sycl` | `kamping::v2::<name>` | Ecosystem adapters (load `v2`), if enabled when installing |

Without `COMPONENTS`, all installed components are loaded. The package bundles kamping-types.

### Building the tests and examples

```bash
cmake --preset debug                     # or: release, relwithdeb
cmake --build --preset debug --parallel
ctest --preset debug
```

| CMake option | Default | Description |
|--------------|---------|-------------|
| `KAMPING_V2_BUILD_EXAMPLES_AND_TESTS` | `OFF` | Build tests and examples (on in the presets) |
| `KAMPING_V2_WARNINGS_ARE_ERRORS` | `OFF` | Treat warnings as errors in tests and examples |
| `KAMPING_V2_INSTALL` | on if top-level | Generate install rules |
| `KAMPING_V2_ENABLE_REFLECTION` | `ON` | Derive MPI datatypes for plain structs via Boost.PFR |
| `KAMPING_V2_REFLECTION_USE_SYSTEM_BOOST_FOR_PFR` | `OFF` | Take Boost.PFR from a system Boost |
| `KAMPING_V2_ENABLE_CEREAL` | `OFF` | Build the Cereal adapter |
| `KAMPING_V2_ENABLE_KOKKOS` | `OFF` | Build the Kokkos adapter |
| `KAMPING_V2_ENABLE_KOKKOS_COMM` | `OFF` | Let the Kokkos adapter use KokkosComm |
| `KAMPING_V2_ENABLE_THRUST` | `OFF` | Build the Thrust adapter (CUDA) |
| `KAMPING_V2_ENABLE_SYCL` | `OFF` | Build the SYCL adapter (Intel oneAPI DPC++) |

## Publications

KaMPIng v2 and its core interface are described in

> T. N. Uhl, M. Schimek, D. Brommer. **Concepts in Practice: C++ MPI Bindings for the HPC Ecosystem — From a
> Standardizable Core to a Composable Interface.** In *Recent Advances in the Message Passing Interface (EuroMPI
> 2026)*, LNCS, Springer. [doi:10.1007/978-3-032-40777-1_2](https://doi.org/10.1007/978-3-032-40777-1_2) ·
> preprint: [arXiv:2606.09102](https://arxiv.org/abs/2606.09102)

which realizes the design principles of

> C. N. Avans, A. A. Correa, S. Ghosh, M. Schimek, J. Schuchart, A. Skjellum, E. D. Suggs, T. N. Uhl. **Concepts for
> Designing Modern C++ Interfaces for MPI.** In *EuroMPI 2025*, LNCS, pp. 165–183, Springer.
> [doi:10.1007/978-3-032-07194-1_10](https://doi.org/10.1007/978-3-032-07194-1_10) ·
> preprint: [arXiv:2506.14610](https://arxiv.org/abs/2506.14610)

If you use KaMPIng v2, please cite the first paper:

```bibtex
@inproceedings{UhlSB26,
  author        = {Tim Niklas Uhl and Matthias Schimek and Daniel Brommer},
  title         = {Concepts in Practice: {C++} {MPI} Bindings for the {HPC}
                   Ecosystem --- From a Standardizable Core to a Composable
                   Interface},
  booktitle     = {Recent Advances in the Message Passing Interface
                   (EuroMPI 2026)},
  series        = {Lecture Notes in Computer Science},
  publisher     = {Springer},
  year          = {2026},
  doi           = {10.1007/978-3-032-40777-1_2},
  eprint        = {2606.09102},
  archivePrefix = {arXiv}
}
```

## License

KaMPIng v2 is released under the [Boost Software License 1.0](LICENSE).
