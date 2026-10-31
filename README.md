<p align="center">
  <img src="docs/icon.svg" alt="KaMPIng logo" width="140">
</p>

<h1 align="center">KaMPIng v2</h1>

<p align="center">
  <em>MPI for modern C++: pass your containers, let the library figure out the rest.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/kamping-site/kamping-v2" alt="License">
</p>

---

MPI is powerful, but calling it from C++ means juggling raw pointers, element counts, datatypes, and
displacement arrays by hand, and getting any of them wrong usually crashes or hangs. **KaMPIng v2** lets you
hand MPI the objects you already have, like a `std::vector`, a `std::span`, or a Kokkos view. The library works
out the pointer, the count, and the datatype for you. When you need more, such as a receive buffer that grows to
fit, per-rank counts, or a custom datatype, you add it with a small, readable `|` pipe, much like
`std::ranges` views.

```cpp
#include <kamping/v2.hpp>

namespace v2 = kamping::v2;

int main(int argc, char* argv[]) {
    v2::environment env(argc, argv);
    v2::comm_view   comm{MPI_COMM_WORLD};

    // Point-to-point: the receiver does not need to know the message size in advance.
    if (comm.rank() == 0) {
        std::vector<int> data{1, 2, 3, 4};
        v2::send(data, 1, comm);
    } else if (comm.rank() == 1) {
        std::vector<int> data;
        v2::recv(data | v2::views::resize, 0, comm); // resized to fit the message
    }

    // Collectives: every rank contributes a different amount; counts are exchanged for you.
    std::vector<int> mine(static_cast<std::size_t>(comm.rank()) + 1, comm.rank());
    auto [_, everything] = v2::allgatherv(mine, v2::auto_recv_v<int>(), comm);

    // Reductions with standard functors; ref_single turns a single value into a buffer.
    int rank = comm.rank();
    int sum  = 0;
    v2::allreduce(v2::views::ref_single(rank), v2::views::ref_single(sum), std::plus<>{}, comm);
}
```

KaMPIng v2 is header-only and requires C++20. It is the successor of [KaMPIng](https://github.com/kamping-site/kamping),
redesigned around C++20 concepts instead of named parameters.

## Highlights

- **Zero-boilerplate buffers.** Any contiguous range of a builtin type is a valid MPI buffer as is. Other types
  plug in through a trait or a few member functions, with no inheritance and no wrappers.
- **Receive buffers that size themselves.** Pipe a container through `views::resize`, and KaMPIng probes the
  message or exchanges the counts before receiving.
- **Composable views.** `with_type`, `with_counts`, `with_displs`, `auto_displs`, `flatten_v`, … combine with `|`
  and compute their metadata lazily.
- **Ownership you can see.** Pass an lvalue and the library borrows it. Pass an rvalue and it takes ownership
  and hands the buffer back when the operation completes.
- **Safe non-blocking operations.** `isend`/`irecv` return an `iresult` that keeps its buffer alive. Forgetting
  to complete it is caught loudly instead of silently corrupting memory.
- **Bring your own ecosystem.** Adapters for Cereal (serialization), Kokkos, Thrust, and SYCL.

## How it works

An MPI call needs a few facts about each buffer: where its data starts, how many elements it holds, and their
MPI datatype. Variadic operations such as `alltoallv` also need per-rank counts and displacements. KaMPIng asks
your object for these facts through a small set of accessor functions:

| Accessor | Default for a `std::vector<int>` | Customize via |
|----------|----------------------------------|---------------|
| `count(buf)` | `std::ranges::size` | `mpi_count()` member or `buffer_traits` |
| `ptr(buf)` | `std::ranges::data` | `mpi_ptr()` member or `buffer_traits` |
| `type(buf)` | `MPI_INT` (deduced) | `mpi_type()` member, `buffer_traits`, or `views::with_type(dt)` |
| `counts(buf)`, `displs(buf)` | — | `mpi_counts()`/`mpi_displs()` members, `views::with_counts`, `views::auto_displs`, … |

To adapt a type you can't modify, specialize `buffer_traits`:

```cpp
template <>
struct mpi::experimental::buffer_traits<MyType> {
    static std::ptrdiff_t count(MyType const& t) { return t.size; }
    static int const*     ptr(MyType const& t)   { return t.values; } // used for sending
    static int*           ptr(MyType& t)         { return t.values; } // used for receiving
    static MPI_Datatype   type(MyType const&)    { return MPI_INT; }
};
```

Before each operation, KaMPIng calls a customization point named `infer()`. It fills in whatever a buffer left
open, for example by probing a message to size a `views::resize` buffer. You can overload `infer()` for your
own buffer types to exchange extra metadata.

## Calling conventions

The communicator always comes last. Arguments with an obvious default can be left out from the front of it, so you
only spell out what differs:

```cpp
v2::send(buf, dest, comm);          // tag 0
v2::send(buf, dest, tag, comm);
v2::recv(buf, comm);                // any source, any tag
v2::recv(buf, source, tag, comm);
v2::bcast(buf, comm);               // root 0
v2::reduce(s, r, comm);             // std::plus<>, root 0
v2::reduce(s, r, MPI_MAX, root, comm);
```

Rooted collectives have a shorthand for the ranks that don't need the root-only buffer, e.g. `v2::gather(s, comm)`
on non-root ranks.

## Views

All views compose with `|` and live in `kamping::v2::views`.

| View | What it does |
|------|--------------|
| `resize` | Resizes the container to fit the incoming data before receiving |
| `with_type(dt)` | Uses the MPI datatype `dt` |
| `with_count(n)` | Uses `n` as the element count |
| `with_counts(counts)` | Attaches per-rank counts (variadic operations) |
| `with_displs([monotonic,] displs)` | Attaches per-rank displacements |
| `auto_counts([[resize,] counts])` | Per-rank counts that the operation exchanges for you |
| `auto_displs([[resize,] displs])` | Computes displacements from the counts (exclusive prefix sum) |
| `resize_v` | Resizes a variadic receive buffer to fit its counts and displacements |
| `auto_recv_v` | Shorthand for `auto_counts() \| auto_displs() \| resize_v` |
| `flatten_v()` | Turns a range of per-rank ranges (or `(rank, value)` pairs) into one variadic send buffer |
| `chunked` | The inverse: splits a variadic buffer into per-rank `std::span`s |
| `ref_single(x)` | Treats a single value as a one-element buffer |
| `with_pool(pool)` / `with_auto_pool(pool)` | Attaches a derived datatype from a `type_pool` |

The factories `auto_recv<T>()` and `auto_recv_v<T>()` create a fresh, owned receive buffer that sizes itself.

## Non-blocking operations

Non-blocking operations return an `iresult`, which owns the request and, for rvalue buffers, the buffer as well.
The buffer lives on the heap, so moving the `iresult` around never invalidates the pointer MPI is writing to.

```cpp
// Receiver: the iresult owns the vector until the receive completes.
auto pending = v2::irecv(std::vector<int>{} | v2::views::resize, 0, comm);
// ... do other work ...
std::vector<int> data = pending.wait().underlying(); // the owned buffer is handed back

// Sender: `data` is borrowed, so test() just reports completion.
auto sending = v2::isend(data, 1, comm);
while (!sending.test()) { /* do other work */ }
```

An `iresult` must be completed with `wait()` or a successful `test()`. KaMPIng never completes a request
behind your back. Destroying one that is still active prints a diagnostic and calls `std::terminate()`, so the
buffer is never freed while MPI may still use it.

## Supported operations

**Point-to-point:** `send` (standard, buffered, synchronous, and ready modes), `recv`, `sendrecv`, `isend`, `irecv`,
`isendrecv`, `probe`, `iprobe`, `mprobe`, `improbe`, `mrecv`, `imrecv`

**Collectives:** `allgather`, `allgatherv`, `allreduce`, `alltoall`, `alltoallv`, `barrier`, `ibarrier`, `bcast`,
`exscan`, `gather`, `gatherv`, `reduce`, `scan`, `scatter`, `scatterv`

**Completion:** `wait`, `waitall`, `waitany`, `waitsome`, and their `test` counterparts

**Objects:** communicators, groups, windows, info objects, sessions (MPI 4), attribute caching, and custom
reduction operations via `make_op`

## Getting started

KaMPIng v2 needs CMake 3.25+, a C++20 compiler, and an MPI installation. The easiest way to use it is
`FetchContent`:

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

Include `<kamping/v2.hpp>` for everything, or individual headers such as `<kamping/v2/collectives/alltoallv.hpp>`
to keep compile times down.

### Building the tests and examples

```bash
cmake --preset debug                     # or: release, relwithdeb
cmake --build --preset debug --parallel
ctest --preset debug
```

| CMake option | Default | Description |
|--------------|---------|-------------|
| `KAMPING_BUILD_EXAMPLES_AND_TESTS` | `OFF` | Build tests and examples (on in the presets) |
| `KAMPING_WARNINGS_ARE_ERRORS` | `OFF` | Treat warnings as errors |
| `KAMPING_ENABLE_SERIALIZATION` | `OFF` | Build the Cereal adapter |
| `KAMPING_ENABLE_KOKKOS` | `OFF` | Build the Kokkos adapter |
| `KAMPING_ENABLE_REFLECTION` | `ON` | Derive MPI datatypes for plain structs via Boost.PFR |

## Repository layout

KaMPIng v2 is built in layers. You can use the lower ones on their own.

| Directory | Namespace | CMake target | What it is |
|-----------|-----------|--------------|------------|
| `mpi-core/` | `mpi::experimental` | `mpi::core` | Thin, explicit MPI wrappers: one MPI call each, no inference or resizing |
| `kamping-v2/` | `kamping::v2` | `kamping::v2` | The ergonomic layer: views, `infer()`, self-sizing buffers, `iresult` |
| `dstl/` | `dstl` | `dstl` | Distributed algorithms built on top (sparse all-to-all, grid all-to-all, scans, …) |
| `ecosystem/` | `kamping::v2`, `kamping::v2::views` | `kamping::ecosystem::<name>` | Adapters for Cereal, Kokkos, Thrust, and SYCL |

MPI datatypes for user types come from `kamping-types`, which is shared with
[KaMPIng v1](https://github.com/kamping-site/kamping) and fetched automatically. KaMPIng v1 and v2 are
independent and can be used in the same program.

## License

KaMPIng v2 is released under the [Boost Software License 1.0](LICENSE).
