# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KaMPIng v2 is a **header-only C++20 MPI wrapper library** built on a concept-based buffer protocol with composable range adaptors. It replaces the named-parameter / template-metaprogramming approach of v1 with a clean layered architecture and `std::ranges`-style view pipelines.

## Build Commands

Requires CMake 3.25+ and an MPI installation. Out-of-source builds are enforced.

```bash
# Recommended: use presets
cmake --preset debug
cmake --build --preset debug --parallel

cmake --preset release
cmake --build --preset release --parallel

# Traditional
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DKAMPING_BUILD_EXAMPLES_AND_TESTS=ON
cmake --build build --parallel
```

Key CMake options:
- `KAMPING_BUILD_EXAMPLES_AND_TESTS` — must be ON to build tests (default OFF in library mode)
- `KAMPING_WARNINGS_ARE_ERRORS` — treat compiler warnings as errors (default OFF)
- `KAMPING_ENABLE_SERIALIZATION` — Cereal ecosystem adapter (default OFF)
- `KAMPING_ENABLE_KOKKOS` — Kokkos ecosystem adapter (default OFF)
- `KAMPING_ENABLE_REFLECTION` — Boost.PFR struct reflection in kamping-types (default ON)

## Testing

All tests live under `kamping-v2/tests/`.

```bash
# Run all tests
ctest --preset debug          # or --preset release

# Run a single test by name
ctest --preset debug -R test_<testname>
# Or build and run directly
cmake --build build/debug --target test_<testname>
./build/debug/kamping-v2/tests/test_<testname>
```

MPI tests run with multiple core counts (1, 2, 4, 8).

## Code Formatting

```bash
cmake --build build/debug --target check-clang-format   # validate C++ formatting
cmake --build build/debug --target check-cmake-format   # validate CMake formatting
```

- C++: `clang-format` with `.clang-format` — 120 column limit, 4-space indent
- CMake: `cmake-format` with `.cmake-format.py` — 120 column limit

Run formatters before committing.

## Architecture

### Layered Architecture

v2 is organized into four explicit layers. Each layer depends only on the layers below it.

```
┌─────────────────────────────────────────────────────────┐
│  Ecosystem bridges                                       │
│  Bindings to external libraries (Kokkos, Thrust, …)     │
│  ecosystem/                                              │
├─────────────────────────────────────────────────────────┤
│  Language bindings  (kamping-v2)                         │
│  C++ ergonomics: ownership, infer, deferred buffers,     │
│  auto-counts/displs, resize-on-receive                   │
│  kamping/v2/  (excluding contrib/)                       │
├─────────────────────────────────────────────────────────┤
│  Language bridge                                         │
│  Minimal C++ implementation of the buffer contract:      │
│  count/ptr/type/counts/displs dispatch,                  │
│  core view adaptors (with_counts, with_displs, …),       │
│  mpi::experimental:: MPI wrappers, native_handle bridge  │
│  include/mpi/, kamping/v2/ranges/, kamping/v2/views/     │
├─────────────────────────────────────────────────────────┤
│  Contract  (language-agnostic)                           │
│  Abstract description of send/recv buffer,               │
│  variadic buffer (with counts+displs), native MPI object │
│  — expressed as C++ concepts in ranges/concepts.hpp      │
└─────────────────────────────────────────────────────────┘
```

**Contract layer** — language-agnostic definitions of what a send buffer, recv buffer, variadic buffer (with counts and displacements), and a native MPI handle *are*. Expressed in C++ as the concepts in `kamping/v2/ranges/concepts.hpp` (`send_buffer`, `recv_buffer`, `data_buffer_v`, …). No implementation here — only constraints.

**Language bridge** — the minimal C++ wiring that makes the contract work:
- Accessor dispatch functions (`mpi::experimental::count/ptr/type/counts/displs`) with three-tier priority: `buffer_traits<T>` specialization → member functions (`mpi_count()`, `mpi_ptr()`, etc.) → `std::ranges` / builtin-type fallbacks.
- `mpi::experimental::buffer_traits<T>` and `handle_traits<T>` — non-intrusive customization points for third-party types.
- Ownership infrastructure: `ref_view<T>` (non-owning, wraps lvalue), `owning_view<T>` (owning, wraps rvalue), and `all(r)` / `all_t<R>` which select between them.
- View adaptor machinery: `view_interface_base`, `adaptor_closure`, `adaptor`, `composed_closure` — the pipe `|` operator infrastructure.
- Core view adaptors: `with_type`, `with_size`, `with_counts`, `with_displs`.
- `mpi::experimental::` MPI wrappers — one MPI call each, no inference, no resizing, throw `mpi_error` on failure. Concrete buffer types `mpi_span` and `mpi_span_v`.
- `mpi::experimental::handle` / `handle_ptr` / `to_rank` / `to_tag` — extract raw MPI handles from any wrapper.

**Language bindings (kamping-v2)** — C++ ergonomics and MPI convenience on top of the bridge:
- `infer()` protocol: operation-tagged ADL hook that resolves unknown recv sizes (via `MPI_Mprobe`) or variadic counts before the MPI call is issued.
- Deferred buffer concepts (`deferred_recv_buf`, `deferred_recv_buf_v`) and the views that implement them: `resize`, `resize_v`, `auto_counts`.
- `auto_displs` — computes displacements via `exclusive_scan`; tags result as `has_monotonic_displs` to enable O(1) resize.
- `iresult<Buf>` — move-only non-blocking handle; stores the buffer on the heap via `unique_ptr<all_t<Buf>>` so the pointer captured by MPI remains stable after a move.
- Sentinel buffers (`inplace`, `null_buf`, `bottom`) — zero-overhead special buffer values for collective shortcuts.
- `kamping::v2::` wrappers (send, recv, bcast, …) that call `infer()` then delegate to `mpi::experimental::`.

**Ecosystem bridges** — bindings to external C++ libraries, living in `ecosystem/`. Currently: Kokkos adapter, GPU (Thrust/SYCL) adapters, and Cereal serialization via `views::serialize` / `views::deserialize<T>()`.

### Core Idea: Buffer Protocol + View Pipeline

Callers pipe standard C++ objects through a chain of `std::ranges`-style view adaptors that attach MPI metadata (element count, MPI datatype, per-rank counts, displacements). The resulting view satisfies one of the buffer concepts and is passed directly to a free-function MPI wrapper.

```cpp
kamping::v2::send(v, 1, comm);
kamping::v2::recv(v | kamping::views::resize, comm);           // resizable recv
kamping::v2::send(map | kamping::views::serialize, 1, comm);   // Cereal serialization
```

### Namespaces

| Namespace | Role |
|-----------|------|
| `mpi::experimental::` | Core layer: buffer concepts, accessors (`count`, `ptr`, `type`, `counts`, `displs`), MPI wrappers, native-handle adaption, concrete buffer types (`mpi_span`, `mpi_span_v`, `comm_view`) |
| `kamping::v2::` | High-level wrappers: call `infer()` then delegate to `mpi::experimental::` |
| `kamping::views::` | Range adaptor factory functions (pipe operators) |

### Buffer Concepts (`include/mpi/buffer.hpp`)

| Concept | Requirements |
|---------|-------------|
| `data_buffer` | count + ptr (any pointer) + type |
| `send_buffer` | data_buffer with `ptr()` convertible to `void const*` |
| `recv_buffer` | data_buffer with `ptr()` convertible to `void*` |
| `data_buffer_v` | data_buffer + `counts()` (counts range) + `displs()` (displs range) |
| `deferred_recv_buf` | recv_buffer with `set_recv_count(n)` for late-bound sizes |
| `deferred_recv_buf_v` | variadic version with `set_comm_size`, `mpi_counts()`, `commit_counts()` |

### Accessor Dispatch (`include/mpi/buffer.hpp`)

Each of `count()`, `ptr()`, `type()`, `counts()`, `displs()` is a free function in `mpi::experimental::` with prioritized overload resolution:

1. `buffer_traits<T>` specialization — non-intrusive, for types you don't own
2. `t.mpi_count()` / `t.mpi_ptr()` / `t.mpi_type()` / `t.mpi_counts()` / `t.mpi_displs()` member functions
3. `std::ranges::size` / `std::ranges::data` + builtin MPI type deduction (for non-variadic buffers)

Specialize `mpi::experimental::buffer_traits<T>` to adapt any third-party type without modifying it.

### View Adaptors (`include/kamping/v2/views/`)

All views are composable with `|`. They are lazy; metadata is not computed until the MPI operation queries it.

| View factory | Effect |
|---|---|
| `views::resize` | Marks a container as resizable; MPI recv will call `resize(n)` before writing |
| `views::with_type(dt)` | Overrides the MPI datatype |
| `views::with_size(n)` | Overrides element count |
| `views::with_counts(range)` | Attaches per-rank send/recv counts (variadic operations) |
| `views::with_displs(range)` | Attaches per-rank displacements; pass `kamping::v2::monotonic` tag to enable O(1) resize |
| `views::auto_displs([tag,] [container])` | Computes displacements via exclusive_scan of counts; always `has_monotonic_displs` |
| `views::resize_v` | Variadic recv buffer: resizes the underlying container from counts+displs before receive |
| `views::ref_single(val)` | Wraps a single scalar as a one-element contiguous buffer |
| `views::auto_counts([buf])` | Deferred variadic counts buffer; `set_comm_size` / `commit_counts` protocol |
| `views::serialize` / `views::deserialize<T>()` | Cereal serialization (contrib) |

Ownership semantics follow `std::ranges::all`: an lvalue produces a `ref_view` (borrows); an rvalue produces an `owning_view` (owns). This propagates through the view chain.

### `infer()` Protocol (`include/kamping/v2/infer.hpp`)

Before issuing an MPI call, `kamping::v2::` wrappers call `infer(comm_op::XXX{}, buf..., comm)`. The default overloads:
- For `deferred_recv_buf` targets: probe the network (`MPI_Mprobe`) and call `set_recv_count(n)` so the buffer resizes before the actual receive.
- For variadic operations: call `set_comm_size` and let MPI write directly into the counts buffer.

Users can add new ADL overloads of `infer()` for custom buffer types or to exchange additional metadata.

### Non-Blocking Operations (`include/kamping/v2/iresult.hpp`)

Non-blocking operations return `iresult<Buf>` (single buffer) or `iresult<SBuf, RBuf>` (sendrecv).

- Move-only; the buffer is stored on the heap via `unique_ptr` so the pointer captured by MPI remains stable after a move.
- `.wait([status])` — blocks and returns the buffer (owned buffers moved out; borrowed buffers return an lvalue reference).
- `.test([status])` — polls; borrowed buffers return `bool`, owned buffers return `std::optional<T>`.
- Destructor calls `MPI_Wait` if the request was not already completed, preventing silent data corruption.

### Native Handle Bridge (`mpi-core/include/mpi/handle.hpp`)

Free functions `mpi::experimental::handle(x)` and `handle_ptr(x)` extract `MPI_Comm`, `MPI_Request`, etc. from arbitrary wrapper types. Dispatch priority:

1. `handle_traits<T>` specialization — non-intrusive, for types you don't own
2. `t.mpi_handle()` / `t.mpi_handle_ptr()` member functions
3. Passthrough for raw `MPI_Comm` / `MPI_Request` / … values

The same pattern applies to ranks (`to_rank`) and tags (`to_tag`), supporting both plain `int` and strongly-typed wrappers.

### Coding Conventions

- Header-only: all implementation in `.hpp` files under `include/`
- `#pragma once` for include guards
- East-side `const` (`int const` not `const int`)
- CamelCase for types, snake_case for functions/variables
- Private members prefixed with `_`: `_member`
- Doxygen required for all public API
- No `using namespace` in headers
- `auto` preferred for type deduction

---

## Design Documentation

See **DESIGN.md** for step-by-step patterns for implementing new collectives, views, and handle wrappers.
