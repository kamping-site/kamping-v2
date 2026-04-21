# KaMPIng v2 Design Guide

This document describes the design patterns and layering in KaMPIng v2. It's meant as a reference for implementing new features and understanding the existing code without reading every header.

## Architecture Overview

KaMPIng v2 uses a **layered architecture** with a clear separation of concerns:

```
┌─────────────────────────────────────────────────┐
│  v2 Wrappers (kamping/v2/)                      │
│  - infer() protocol (deferred sizing, metadata) │
│  - Result types (struct binding support)        │
│  - Ownership semantics (ref_view vs owning)     │
└──────────────┬──────────────────────────────────┘
               ↓ delegates to
┌──────────────────────────────────────────────────┐
│  Core Layer (include/mpi/)                      │
│  - Buffer protocol (concepts, accessors)        │
│  - Bare MPI wrappers (one call each)           │
│  - Handle dispatch (traits-based)               │
│  - No view machinery, no resizing               │
└──────────────────────────────────────────────────┘
               ↓ uses
┌──────────────────────────────────────────────────┐
│  View Adapters (kamping/v2/views/)              │
│  - Composable via | operator                    │
│  - Lazy metadata attachment                     │
│  - Ownership forwarding                         │
└──────────────────────────────────────────────────┘
```

The key principle: **core layer is views-free**. All view machinery, ownership forwarding, and resize logic lives in v2 and the views namespace. This keeps the MPI wrappers simple and reusable.

---

## Core Concepts

### 1. Buffer Protocol

All MPI operations work with buffers satisfying one of these concepts:

```cpp
// Single-buffer operations (send, recv, allgather, broadcast, etc.)
concept send_buffer {
    count(t) → std::integral                    // element count
    ptr(t) → convertible to (void const*)       // read-only pointer
    type(t) → MPI_Datatype                      // element type
};

concept recv_buffer {
    count(t) → std::integral
    ptr(t) → convertible to void*               // read-write pointer
    type(t) → MPI_Datatype
};

concept data_buffer = send_buffer || recv_buffer;

// Variadic operations (allgatherv, alltoallv, etc.)
concept send_buffer_v = send_buffer + {
    counts(t) → std::ranges::contiguous_range<int const>  // per-rank counts
    displs(t) → std::ranges::contiguous_range<int const>  // per-rank displacements
};

concept recv_buffer_v = recv_buffer + {
    counts(t) → std::ranges::contiguous_range<int>        // mutable for infer()
    displs(t) → std::ranges::contiguous_range<int const>
};
```

**Accessor dispatch** follows a 3-tier priority:

1. **`mpi::experimental::buffer_traits<T>` specialization** — non-intrusive, for types you don't own
2. **Member functions** — `t.mpi_count()`, `t.mpi_ptr()`, `t.mpi_type()`, `t.mpi_counts()`, `t.mpi_displs()`
3. **Fallback** — `std::ranges::size()`, `std::ranges::data()`, builtin type deduction

To adapt a third-party type:

```cpp
template <>
struct mpi::experimental::buffer_traits<MyBuffer> {
    static std::ptrdiff_t count(MyBuffer const& b) { return b.size; }
    static void* ptr(MyBuffer& b) { return b.data; }  // recv
    static void const* ptr(MyBuffer const& b) { return b.data; }  // send
    static MPI_Datatype type(MyBuffer const& b) { return MPI_INT; }
};
```

### 2. Handle Dispatch

All MPI handles (`MPI_Comm`, `MPI_Request`, `MPI_Datatype`, `MPI_Status`, `MPI_Message`) are extracted via `mpi::experimental::handle(x)` and `mpi::experimental::handle_ptr(x)`, with the same 3-tier dispatch:

1. **`native_handle_traits<T>` specialization**
2. **Member functions** — `t.mpi_handle()`, `t.mpi_handle_ptr()`
3. **Builtin passthrough** — raw `MPI_Comm`, `MPI_Request`, etc. pass through unchanged

Example (non-intrusive wrapper):

```cpp
class MyComm {
    MPI_Comm comm_;
public:
    MPI_Comm get() const { return comm_; }
};

template <>
struct mpi::experimental::native_handle_traits<MyComm> {
    static MPI_Comm handle(MyComm const& c) { return c.get(); }
};

// Now you can pass MyComm directly to mpi::experimental::allgather()
mpi::experimental::allgather(sbuf, rbuf, MyComm{comm});
```

---

## Adding a New Collective

Here's a step-by-step walkthrough using `allgather` as an example.

### Step 1: Implement the Core Wrapper

**File:** `include/mpi/collectives/allgather.hpp`

```cpp
namespace mpi::experimental {

template <
    send_buffer                         SBuf,
    recv_buffer                         RBuf,
    convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void allgather(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) {
    int comm_size = 0;
    MPI_Comm_size(handle(comm), &comm_size);
    KAMPING_ASSERT(count(rbuf) % comm_size == 0, "recv buffer size must be divisible by comm size");
    
    int err = MPI_Allgather(
        ptr(sbuf),
        static_cast<int>(count(sbuf)),
        type(sbuf),
        ptr(rbuf),
        static_cast<int>(count(rbuf)) / comm_size,
        type(rbuf),
        handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

} // namespace mpi::experimental
```

**Key points:**
- Template over buffer concepts (`send_buffer`, `recv_buffer`), not specific types
- Extract metadata via `count()`, `ptr()`, `type()` free functions
- Extract handles via `handle()`
- Use `KAMPING_ASSERT` for precondition validation
- Throw `mpi_error` on MPI failure
- No inference, no resizing, no ownership tricks — pure MPI wrapper

### Step 2: Define the Infer Overload

**File:** `include/kamping/v2/infer.hpp`

The operation tag lives in `comm_op` namespace, and the infer overload is generic:

```cpp
namespace kamping {

// In comm_op namespace (also in same file)
namespace comm_op {
    struct allgather {};
}

// Default infer overload for allgather
template <
    mpi::experimental::send_buffer SBuf,
    mpi::experimental::recv_buffer RBuf>
void infer(comm_op::allgather, SBuf const& sbuf, RBuf& rbuf, MPI_Comm comm) {
    // If recv is a deferred buffer (has set_recv_count), resize it
    if constexpr (kamping::ranges::deferred_recv_buf<RBuf>) {
        int comm_size = 0;
        MPI_Comm_size(comm, &comm_size);
        rbuf.set_recv_count(
            comm_size * static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf))
        );
    }
}

} // namespace kamping
```

**Why a separate infer?** It discovers metadata before the MPI call:
- For deferred buffers: calculate required size from comm_size and send count
- For variadic ops: negotiate counts/displacements via collective calls
- For recv: probe network to discover incoming message size

The infer is called by the v2 wrapper before delegating to the core layer.

### Step 3: Implement the v2 Wrapper

**File:** `include/kamping/v2/collectives/allgather.hpp`

```cpp
namespace kamping::v2 {

template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto allgather(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD)
    -> result<SBuf, RBuf> {
    
    // Construct result, preserving ownership (lvalue → reference, rvalue → owned)
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    
    // Run infer to discover/resize metadata before MPI call
    infer(comm_op::allgather{}, res.send, res.recv, mpi::experimental::handle(comm));
    
    // Delegate to core layer
    mpi::experimental::allgather(res.send, res.recv, comm);
    
    return res;
}

} // namespace kamping::v2
```

**Pattern:**
1. Construct `result<SBuf, RBuf>` wrapper (ownership preservation handled automatically)
2. Call `infer(comm_op::allgather{}, ...)` to discover/resize metadata
3. Delegate to `mpi::experimental::allgather()`
4. Return result (supports structured bindings: `auto [s, r] = allgather(...)`)

### Step 4: Tests

Create `tests/v2/v2_allgather_test.cpp` with:
- Single-rank send/recv
- Multi-rank with uniform sizes
- Multi-rank with non-uniform send sizes (validate recv sizing)
- Structured binding usage
- Deferred buffers (with `views::resize`)

---

## Working with Views

### The View Interface Pattern

All v2 views derive from `view_interface<Derived>` and implement a `base()` method:

```cpp
template <typename Base>
struct resize_view : public view_interface<resize_view<Base>> {
    Base base_;  // The wrapped range
    
    auto base() { return base_; }
    auto base() const { return base_; }
};
```

`view_interface` is a **CRTP mixin** that:
- Forwards `begin()`, `end()` to `base()`
- Forwards MPI accessors (`mpi_count()`, `mpi_ptr()`, `mpi_type()`) via `mpi::experimental::*()` dispatch
- Conditionally provides `mpi_resize_for_receive()` if base supports `resize`

This means a view automatically satisfies `recv_buffer` if its base does, even if the view itself doesn't define these methods.

### Composing Views with `|`

```cpp
// Each view is an adaptor_closure that supports piping
std::vector<int> data = {...};

// Composed view: resize on receive, override element type
auto view = data | kamping::views::resize | kamping::views::with_type(MPI_INT);

// Lazy: metadata is computed only when MPI operation queries it
// resize is forwarded through the adaptor chain via mpi_resize_for_receive()
```

**Implementation sketch:**

```cpp
namespace kamping::views {

struct resize_adaptor : adaptor_closure<resize_adaptor> {
    template <std::ranges::range R>
    auto operator()(R&& r) const {
        return resize_view<all_t<R>>{all(std::forward<R>(r))};
    }
};

inline constexpr resize_adaptor resize;

} // namespace kamping::views
```

The `adaptor_closure` base provides `operator|` overloads that call `operator()`.

---

## Variadic Operations (counts/displacements)

Variadic collectives (`allgatherv`, `alltoallv`, etc.) require `send_buffer_v` / `recv_buffer_v`:

```cpp
template <send_buffer_v SBuf, recv_buffer_v RBuf, ...>
void allgatherv(SBuf&& sbuf, RBuf&& rbuf, MPI_Comm comm) {
    // Metadata: per-rank counts and displacements
    auto counts_s = counts(sbuf);  // std::span<int const>
    auto displs_s = displs(sbuf);  // std::span<int const>
    auto counts_r = counts(rbuf);  // std::span<int> (may be mutable for infer)
    auto displs_r = displs(rbuf);  // std::span<int const>
    
    MPI_Allgatherv(
        ptr(sbuf), counts_s.data(), displs_s.data(), type(sbuf),
        ptr(rbuf), counts_r.data(), displs_r.data(), type(rbuf),
        comm
    );
}
```

**View support:**
- `views::with_counts(range)` — attaches per-rank counts
- `views::with_displs(range)` — attaches per-rank displacements
- `views::auto_counts()` — deferred counts; `infer()` fills in via `set_comm_size` + `commit_counts()`
- `views::auto_displs([container])` — computes displs from counts via exclusive_scan

---

## Non-Blocking Operations

Non-blocking MPI calls return `iresult<Buf>` (or `iresult<SBuf, RBuf>` for sendrecv):

```cpp
// Core layer: raw MPI
template <send_buffer SBuf, ...>
MPI_Request isend(SBuf&& sbuf, int dest, int tag, MPI_Comm comm) {
    MPI_Request req;
    THROW_IF_MPI_ERROR(
        MPI_Isend(ptr(sbuf), count(sbuf), type(sbuf), dest, tag, comm, &req)
    );
    return req;
}

// v2 layer: RAII + buffer ownership
template <send_buffer SBuf, ...>
auto v2::isend(SBuf&& sbuf, int dest, int tag, MPI_Comm comm)
    -> iresult<SBuf> {
    infer(comm_op::isend{}, sbuf, ...);
    MPI_Request req = mpi::experimental::isend(sbuf, dest, tag, comm);
    
    // Store buffer on heap so it survives a move
    return iresult<SBuf>{req, std::make_unique<all_t<SBuf>>(all(std::forward<SBuf>(sbuf)))};
}
```

**`iresult<Buf>` properties:**
- Move-only (buffer pointer captured by MPI stays stable)
- `.wait()` — blocks until completion, returns buffer (owned buffers moved out; borrowed buffers return reference)
- `.test()` — non-blocking poll
- Destructor calls `MPI_Wait` if not already completed (prevents silent hangs)

---

## Deferred Buffers

Deferred buffers are used when sizes aren't known until communication time (e.g., `MPI_Recv` without a prior `MPI_Probe`).

```cpp
concept deferred_recv_buf<T> = recv_buffer<T> && requires(T& t) {
    t.set_recv_count(std::ptrdiff_t(0));  // Called by infer()
};
```

**Example: `views::resize`**

```cpp
template <std::ranges::range Base>
struct resize_view : view_interface<resize_view<Base>> {
    Base base_;
    
    void mpi_resize_for_receive(std::ptrdiff_t n) {
        // Call resize(n) on the underlying container if it has one
        if constexpr (std::ranges::sized_range<Base> && requires { base_.resize(n); }) {
            base_.resize(n);
        }
    }
};
```

When `infer(comm_op::recv{}, buf | views::resize, ...)` is called:
1. Detects that the view is deferred
2. Probes the network via `MPI_Mprobe` to determine size
3. Calls `mpi_resize_for_receive(n)` to resize the buffer
4. Stores the matched message in `infer()` so the actual recv uses it

---

## Extensibility

### Custom Buffer Types

Implement the buffer protocol and optionally `set_recv_count()`:

```cpp
struct MyBuffer {
    void* data;
    std::ptrdiff_t size;
    MPI_Datatype dtype;
    
    void resize(std::ptrdiff_t n) { size = n; /* ... */ }
    void set_recv_count(std::ptrdiff_t n) { resize(n); }
    
    // Optional: provide intrusive accessors
    std::ptrdiff_t mpi_count() const { return size; }
    void* mpi_ptr() { return data; }
    MPI_Datatype mpi_type() const { return dtype; }
};

// Or use buffer_traits for non-intrusive customization
template <>
struct mpi::experimental::buffer_traits<MyBuffer> { /* ... */ };
```

### Custom MPI Handles

Implement `mpi_handle()` or specialize `native_handle_traits`:

```cpp
class CommWrapper {
    MPI_Comm comm_;
public:
    MPI_Comm mpi_handle() const { return comm_; }
};

// Or non-intrusively:
template <>
struct mpi::experimental::native_handle_traits<CommWrapper> {
    static MPI_Comm handle(CommWrapper const& c) { return c.get_comm(); }
};
```

### Custom Infer Overloads

Add ADL-discovered `infer()` overloads for custom types or operations:

```cpp
namespace my_namespace {
    struct CustomOp {};
    
    template <typename SBuf, typename RBuf>
    void infer(comm_op::CustomOp, SBuf& sbuf, RBuf& rbuf, MPI_Comm comm) {
        // Custom size discovery logic
        // Resize rbuf if deferred
    }
}

// Called automatically before MPI operation when using a compatible buffer
```

---

## Testing Patterns

**Test structure:**

```cpp
#include <gtest/gtest.h>
#include <kamping/v2/collectives/allgather.hpp>
#include <mpi.h>

class AllgatherTest : public ::testing::Test {
protected:
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    
    void SetUp() override {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }
};

TEST_F(AllgatherTest, UniformSendSizes) {
    std::vector<int> send_data(10);
    std::vector<int> recv_data(10 * size);
    
    auto [s, r] = kamping::v2::allgather(send_data, recv_data, comm);
    
    EXPECT_EQ(r.size(), 10 * size);
    // Verify contents...
}

TEST_F(AllgatherTest, DeferredRecvBuffer) {
    std::vector<int> send_data = {rank};
    std::vector<int> recv_data;  // Deferred size
    
    auto [s, r] = kamping::v2::allgather(
        send_data,
        recv_data | kamping::views::resize,
        comm
    );
    
    EXPECT_EQ(r.size(), size);  // Should be resized by infer()
}
```

**Run with multiple rank counts:**

```bash
ctest --test-dir build -R test_v2_allgather  # Runs with 1, 2, 4, 8 ranks
```

---

## Quick Reference: Where to Add Things

| Feature | Files | Key Classes/Functions |
|---------|-------|----------------------|
| New collective | `include/mpi/collectives/*.hpp` (core), `include/kamping/v2/collectives/*.hpp` (v2), `include/kamping/v2/infer.hpp` (infer tag + overload) | `mpi::experimental::foo()`, `comm_op::foo`, `infer(comm_op::foo, ...)` |
| New view adaptor | `include/kamping/v2/views/*.hpp` | Derive from `view_interface<Derived>`, implement `base()` |
| Custom buffer | User code | Specialize `buffer_traits<T>` or provide `mpi_count()`, `mpi_ptr()`, `mpi_type()` |
| Custom handle | User code | Specialize `native_handle_traits<T>` or provide `mpi_handle()` |
| Non-blocking variant | Same files as blocking | Return `iresult<Buf>` instead of `void` |

---

## See Also

- `CLAUDE.md` — project structure, build commands, code conventions
- `include/kamping/v2/TODO.md` — remaining work and design decisions
