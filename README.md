![GitHub](https://img.shields.io/github/license/kamping-site/kamping-v2)

# KaMPIng v2

KaMPIng v2 is a next-generation C++20 MPI wrapper built around a *concept-based buffer protocol* and composable range adaptors. Instead of wrapping arguments in named-parameter factories, callers pipe standard C++ objects through `std::ranges`-style view chains that attach MPI metadata (element count, MPI datatype, per-rank counts, displacements). The resulting view satisfies one of the buffer concepts and is passed directly to a free-function MPI wrapper.

```cpp
// Simple point-to-point
kamping::v2::send(data, 1, comm);
kamping::v2::recv(data | kamping::views::resize, comm);

// Variadic collective with automatic counts/displacements
kamping::v2::alltoallv(send_buf | kamping::views::with_counts(scounts),
                       recv_buf | kamping::views::resize_v | kamping::views::auto_counts(),
                       comm);

// Non-blocking with automatic resize
auto req = kamping::v2::irecv(buf | kamping::views::resize, comm);
// ... other work ...
auto result = req.wait();
```

## Repository layout

| Component | Description | C++ standard | CMake target |
|-----------|-------------|:---:|---|
| `mpi-core/` | Low-level MPI wrappers. Buffer concepts and accessor dispatch (`mpi::experimental::`). | C++20 | `kamping::mpi_core` |
| `kamping-v2/` | High-level wrappers with ownership, infer, resize, auto-counts/displs (`kamping::v2::`). | C++20 | `kamping::v2` |
| `ecosystem/cereal/` | Cereal serialization adapter (`kamping::ecosystem::cereal`). | C++20 | `kamping::ecosystem::cereal` |

`kamping-types` (MPI datatype registry, `kamping::types`) is consumed from the [kamping-site/kamping](https://github.com/kamping-site/kamping) repository via FetchContent.

## Quick Start

### Using kamping-v2 via FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    kamping_v2
    GIT_REPOSITORY https://github.com/kamping-site/kamping-v2.git
    GIT_TAG main
)
FetchContent_MakeAvailable(kamping_v2)
target_link_libraries(myapp PRIVATE kamping::v2)
```

### Using only mpi-core via FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    kamping_v2
    GIT_REPOSITORY https://github.com/kamping-site/kamping-v2.git
    GIT_TAG main
    SOURCE_SUBDIR mpi-core
)
FetchContent_MakeAvailable(kamping_v2)
target_link_libraries(myapp PRIVATE kamping::mpi_core)
```

## Building

Requires CMake 3.25+, a C++20 compiler, and an MPI installation.

```bash
# Configure (builds tests and examples)
cmake --preset debug    # or: release, relwithdeb

# Build
cmake --build --preset debug --parallel

# Test
ctest --preset debug
```

Key CMake options:

| Option | Default | Description |
|--------|---------|-------------|
| `KAMPING_BUILD_EXAMPLES_AND_TESTS` | `OFF` | Build tests and examples |
| `KAMPING_WARNINGS_ARE_ERRORS` | `OFF` | Treat warnings as errors |
| `KAMPING_ENABLE_SERIALIZATION` | `ON` | Build the Cereal ecosystem adapter |
| `KAMPING_ENABLE_REFLECTION` | `ON` | Enable Boost.PFR struct reflection in kamping-types |

## Architecture

v2 is organized into four explicit layers. Each layer depends only on the layers below it.

```
┌─────────────────────────────────────────────────┐
│  Ecosystem bridges                              │
│  Bindings to external libraries (Cereal, …)     │
│  ecosystem/                                     │
├─────────────────────────────────────────────────┤
│  Language bindings  (kamping-v2)                │
│  Ownership, infer, deferred buffers,            │
│  auto-counts/displs, resize-on-receive          │
│  kamping-v2/include/kamping/v2/                 │
├─────────────────────────────────────────────────┤
│  Language bridge  (mpi-core)                    │
│  Buffer concepts and accessor dispatch,         │
│  core view adaptors, MPI wrappers,              │
│  native-handle bridge                           │
│  mpi-core/include/mpi/                          │
├─────────────────────────────────────────────────┤
│  Contract  (language-agnostic)                  │
│  Buffer and native-handle concepts              │
└─────────────────────────────────────────────────┘
```

### Buffer Protocol

Any type satisfying the buffer concepts can be passed to MPI wrappers — no inheritance required. Accessor dispatch follows a three-tier priority:

1. `mpi::experimental::buffer_traits<T>` specialization (non-intrusive, for types you don't own)
2. `t.mpi_count()` / `t.mpi_ptr()` / `t.mpi_type()` member functions
3. `std::ranges::size` / `std::ranges::data` + builtin MPI type deduction

```cpp
// Adapt any third-party type non-intrusively:
template <>
struct mpi::experimental::buffer_traits<MyType> {
    static std::ptrdiff_t count(MyType const& t) { return 1; }
    static int const*     ptr(MyType const& t)   { return &t.val; }
    static int*           ptr(MyType& t)         { return &t.val; }
    static MPI_Datatype   type(MyType const&)    { return MPI_INT; }
};
```

### View Adaptors

All views are composable with `|` and lazy:

| View factory | Effect |
|---|---|
| `views::resize` | Marks a container as resizable; MPI recv resizes before writing |
| `views::with_type(dt)` | Overrides the MPI datatype |
| `views::with_size(n)` | Overrides element count |
| `views::with_counts(range)` | Attaches per-rank send/recv counts (variadic operations) |
| `views::with_displs(range)` | Attaches per-rank displacements |
| `views::auto_displs([tag])` | Computes displacements via exclusive_scan of counts |
| `views::resize_v` | Variadic recv buffer: resizes from counts+displs before receive |
| `views::auto_counts([buf])` | Deferred variadic counts buffer |
| `views::serialize` / `views::deserialize<T>()` | Cereal serialization (ecosystem/cereal) |

### Non-Blocking Operations

Non-blocking calls return `iresult<Buf>` — a move-only handle that stores the buffer on the heap so the pointer captured by MPI stays stable. The destructor calls `MPI_Wait` if not already completed.

```cpp
auto req = kamping::v2::isend(data, 1, comm);
// ... other work ...
req.wait();

auto req2 = kamping::v2::irecv(buf | kamping::views::resize, comm);
auto result = req2.wait();  // returns the resized buffer
```

## Relation to KaMPIng v1

The original KaMPIng (named-parameter / template-metaprogramming API, C++17) lives in [kamping-site/kamping](https://github.com/kamping-site/kamping). v1 and v2 share the `kamping-types` type-registry library (maintained in the v1 repo). The two APIs are independent and can coexist in the same binary.

## License

KaMPIng-v2 is released under the Boost Software License 1.0. See [LICENSE](LICENSE) for details.
