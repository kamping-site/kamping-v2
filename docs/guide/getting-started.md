# Getting started

## Adding KaMPIng-v2 to your project

KaMPIng-v2 is header-only. It needs a C++20 compiler, CMake 3.25 or newer and an MPI installation; it is tested with
MPICH, Open MPI and Intel MPI, using GCC and Clang. The simplest way to use it is `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(
    kamping-v2
    GIT_REPOSITORY https://github.com/kamping-site/kamping-v2.git
    GIT_TAG main
)
FetchContent_MakeAvailable(kamping-v2)

target_link_libraries(my_app PRIVATE kamping::v2)
```

To use an ecosystem adapter, set its option before `FetchContent_MakeAvailable`, e.g.
`set(KAMPING_V2_ENABLE_KOKKOS ON)`, and link its target.

### As an installed package

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKAMPING_V2_ENABLE_CEREAL=ON
cmake --install build --prefix /path/to/prefix
```

```cmake
find_package(kamping-v2 REQUIRED COMPONENTS v2 cereal)
target_link_libraries(my_app PRIVATE kamping::v2 kamping::v2::cereal)
```

| Component | Target | Contents | Option |
|-----------|--------|----------|--------|
| `v2` | `kamping::v2` | KaMPIng-v2 (loads `core`) | |
| `dstl` | `dstl::dstl` | Distributed algorithms (loads `v2`) | |
| `cereal` | `kamping::v2::cereal` | Cereal serialization | `KAMPING_V2_ENABLE_CEREAL` |
| `kokkos` | `kamping::v2::kokkos` | Kokkos views | `KAMPING_V2_ENABLE_KOKKOS` |
| `thrust` | `kamping::v2::thrust` | Thrust device vectors (CUDA) | `KAMPING_V2_ENABLE_THRUST` |
| `sycl` | `kamping::v2::sycl` | SYCL buffers (Intel oneAPI DPC++) | `KAMPING_V2_ENABLE_SYCL` |

Adapters are installed if they were enabled when installing. Without `COMPONENTS`, all installed components are
loaded.

## A first program

Everything is available through a single header; individual headers such as
`<kamping/v2/collectives/alltoallv.hpp>` keep compile times down. The examples in this documentation use two
namespace aliases:

```{literalinclude} ../../kamping-v2/examples/getting_started_example.cpp
:language: cpp
:start-after: "// [docs:include]"
:end-before: "// [docs:include]"
```

{cpp:class}`kamping::v2::environment` initializes MPI and finalizes it when it goes out of scope.
{cpp:class}`kamping::v2::comm_view` is a non-owning view of a communicator; any function that expects a communicator
also accepts a plain `MPI_Comm`.

```{literalinclude} ../../kamping-v2/examples/getting_started_example.cpp
:language: cpp
:start-after: "// [docs:setup]"
:end-before: "// [docs:setup]"
:dedent: 4
```

### Sending and receiving

A `std::vector<int>` is a *buffer*: KaMPIng-v2 knows its data pointer, its size and that its elements are sent as
`MPI_INT`. On the receiving side, the size of the message is often not known in advance. Piping the vector through
{cpp:any}`views::resize <kamping::v2::views::resize>` tells the receive to probe the message first and resize the
vector to fit:

```{literalinclude} ../../kamping-v2/examples/getting_started_example.cpp
:language: cpp
:start-after: "// [docs:send-recv]"
:end-before: "// [docs:send-recv]"
:dedent: 4
```

Without `views::resize`, `recv` receives into the vector as it is and never communicates beyond the single
`MPI_Recv`. This is the general rule in KaMPIng-v2: convenience that costs something, such as a probe or an extra
count exchange, only happens if you ask for it at the call site.

### Variable-size collectives

In an `allgatherv`, every rank contributes a different number of elements. With plain MPI, you first exchange the
counts, compute displacements with a prefix sum and allocate the receive buffer. {cpp:any}`kamping::v2::auto_recv_v`
creates a receive buffer that does all three for you:

```{literalinclude} ../../kamping-v2/examples/getting_started_example.cpp
:language: cpp
:start-after: "// [docs:allgatherv]"
:end-before: "// [docs:allgatherv]"
:dedent: 4
```

`auto_recv_v<int>()` is shorthand for a pipeline of three views, which you can also spell out and combine with your
own storage:

```cpp
std::vector<int> all;
v2::allgatherv(mine, all | views::auto_counts | views::auto_displs | views::resize_v, world);
```

See {doc}`../reference/views` for all views.

## Next steps

- {doc}`buffers` explains what makes a type a buffer and how to adapt your own types.
- {doc}`views` and {doc}`receive-buffers` show how to attach counts, datatypes and resizing.
- {doc}`nonblocking` covers non-blocking operations and buffer ownership.
