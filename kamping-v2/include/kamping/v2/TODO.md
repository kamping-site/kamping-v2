# v2 TODO

## Cross-cutting: error handling (`std::expected`)

Tracked in issue #781.

### Context

`mpi::experimental::` wrappers always throw `mpi_error` on failure. This is correct for
programs that use the default `MPI_ERRORS_ARE_FATAL` error handler. Non-throwing error
handling matters primarily for **ULFM** (User-Level Failure Mitigation) where process
failures are expected and handled rather than fatal.

### The owned-buffer problem

When a user moves a buffer into an operation and the MPI call fails, the buffer has been
consumed by the call site but the data was not sent. The error return must carry the buffer
back so the caller can inspect or retry:

```cpp
// Without nothrow: buffer is gone if send throws
v2::send(std::move(vec), dest, comm);

// With nothrow: buffer is returned alongside the error so the caller can recover
auto result = v2::try_send(std::move(vec), dest, comm);
// result.error() carries both mpi_error and the buffer on failure
// result.value() carries the buffer on success (so it can be reused)
```

This means the error type is not a plain `mpi_error` — it must bundle the owned buffers.

### API shape

**Tag dispatch (preferred over suffix naming):**
```cpp
v2::send(buf, dest, comm);                   // throws on error
v2::send(buf, dest, comm, kamping::v2::nothrow); // returns std::expected
```
Mirrors `std::nothrow_t` / `std::allocator` convention. Same function name, overload on the
trailing tag. Avoids a proliferation of `try_*` names while keeping call sites readable.

**Return type when owned buffers are present:**

`std::expected<T, E>` requires T *or* E — not both. Since we always return the buffer
(success or failure), we need a custom error type:

```cpp
template <typename... OwnedBufs>
struct mpi_error_with_bufs {
    mpi_error          error;
    std::tuple<OwnedBufs...> bufs;   // buffers returned on failure so callers can recover
};

// send with owned sbuf:
std::expected<SBuf, mpi_error_with_bufs<SBuf>>
// allreduce with owned sbuf + rbuf:
std::expected<RBuf, mpi_error_with_bufs<SBuf, RBuf>>
// recv with no owned sbuf, owned rbuf:
std::expected<RBuf, mpi_error_with_bufs<RBuf>>
```

If no buffer is owned (all borrows), the error type reduces to plain `mpi_error`:
```cpp
std::expected<void, mpi_error>
```

**For non-blocking:** `iresult::try_wait()` → `std::expected<T, mpi_error_with_bufs<...>>`
is the primary first target since ULFM use cases most often deal with in-flight requests.

### Implementation notes

- `mpi::experimental::` layer: always throws. It is minimal wiring, not a user API.
  Non-throwing paths live only in `kamping::v2::`.
- The `nothrow` overloads call `mpi::experimental::` inside a `try/catch`, extract the
  buffer from the view chain before rethrowing, and pack both into the expected error.
- Both send and recv owned buffers are returned in the error case. Recv buffer contents
  may be garbage (partial write), but returning it avoids a memory leak and lets the
  caller decide whether to inspect or discard it.
- **Defer until ULFM work begins.** Record the design now; implement when needed.

---

## Cross-cutting: large element counts (MPI-4 `_c` variants)

Tracked in issue #113.

### Context

MPI-4 added `_c`-suffixed variants (`MPI_Send_c`, `MPI_Allreduce_c`, …) that accept
`MPI_Count` (64-bit signed) instead of `int` for element counts. Required for buffers
with more than 2^31 elements. MPI-3 and earlier are limited to `int` counts.

### Decision

**Accessor API is unchanged.** `count()` continues to return `std::ptrdiff_t`, which is
already 64-bit on all targets v2 supports. The 32-bit limitation lives only at the MPI
call site, not in the buffer protocol.

**In `mpi::experimental::` wrappers**, dispatch at compile time on MPI version:

```cpp
#if MPI_VERSION >= 4
    MPI_Allreduce_c(ptr(sbuf), ptr(rbuf), count(sbuf), type(sbuf), mpi_op, comm);
#else
    KASSERT(count(sbuf) <= INT_MAX, "element count exceeds int range; requires MPI-4");
    MPI_Allreduce(ptr(sbuf), ptr(rbuf), static_cast<int>(count(sbuf)), type(sbuf), mpi_op, comm);
#endif
```

The `#if` is compiled away — zero overhead, no runtime branch. On MPI < 4 the `KASSERT`
fires in debug builds when a count overflows `int` (uses the existing
`KAMPING_ASSERTION_LEVEL` infrastructure).

**No new CMake option.** MPI version is already detected by `find_package(MPI)`.

**`MPI_Count` vs `std::ptrdiff_t`:** both are 64-bit signed on all common targets;
`static_cast<MPI_Count>(count(buf))` is always safe. No ABI concern.

**When to implement:** in a single dedicated pass over all wrappers after the wrapper set
is complete. Adding guards wrapper-by-wrapper risks inconsistency and makes the diff
harder to review. Open a dedicated issue/PR for the MPI-4 large-count pass.

---

## Environment / Session

### Info object (`include/mpi/info.hpp`)

- [x] **`mpi::experimental::info`** — owning RAII wrapper for `MPI_Info`.
  Adapted from PR #784 (v1 `kamping::Info`) with these fixes applied:
  - Move assignment returns `info&` (not `Info` by value)
  - `nth_key`: uses `char[MPI_MAX_INFO_KEY+1]` buffer; `std::string(buf)` truncates at first `\0`
  - Iterator replaced with C++20 sentinel design: `entry_iterator` (stores `_info + _idx`) +
    `entry_sentinel` (stores `nkeys`); `operator==` is a single comparison; `entries()` returns
    `std::ranges::subrange<entry_iterator, entry_sentinel>` for range-algorithm compatibility
  - `mpi_handle() const → MPI_Info` exposed; `MPI_Info` added to `builtin_handle` concept
  - `info_value_traits<T>` with built-in specializations for `std::string`, `bool`, and
    all non-bool integral types (C++20 requires-clause instead of enable_if SFINAE)

- [x] **`mpi::experimental::info_view`** — non-owning wrapper (mirrors `comm_view`); wraps
  an existing `MPI_Info` without freeing it. Used when passing `MPI_INFO_NULL` or a borrowed handle.

- [x] **`ThreadLevel` enum** (`include/mpi/thread_level.hpp`) — `mpi::experimental::ThreadLevel`
  with `<=>` ordering; re-exported as `kamping::v2::ThreadLevel`. `info_value_traits<ThreadLevel>`
  specialization deferred to the Info/Session work.

### `v2::environment` (world-model initialization)

- [x] **`v2::environment`** (`include/kamping/v2/environment.hpp`) — RAII wrapper for
  `MPI_Init_thread` / `MPI_Finalize`. Non-copyable, non-movable. Destructor guards against
  double-finalization. Explicit `finalize()` for error-handling use cases. Static `initialized()`,
  `finalized()`, `thread_level()` queries. Utility methods (wtime, tag bounds, etc.) left out
  intentionally — those belong elsewhere in v2.

### `v2::session` (MPI-4 sessions model)

- [ ] **`v2::session`** — owning RAII wrapper for `MPI_Session`. Move-only (delete copy).
  Draft exists in PR #772; adapt with these fixes:
  - Destructor must guard against calling `MPI_Session_finalize` after MPI is already finalized
  - Make move-only: delete copy constructor/assignment; define move constructor/assignment
  - `pset_name_is_valid` KASSERT is O(N psets × MPI calls) — replace with a cheap syntactic
    check (non-empty, valid prefix) rather than enumerating all psets
  - Expose psets as a lazy range via `session::psets() → /*range of string*/` rather than
    requiring the caller to manage `begin`/`end` with an `Info` argument

  **Ergonomic API:**

  ```cpp
  kamping::v2::session session;

  // Shortest form — no group name needed at all
  auto comm = session.comm_from_pset(psets::world);

  // Group as a short-lived temporary — freed immediately after MPI_Comm_create_from_group returns
  auto comm = v2::comm(session.group_from_pset(psets::world));

  // Named group — only when you need it for multiple comms or group queries
  auto group = session.group_from_pset(psets::world);
  auto comm1 = v2::comm(group, "tag-a");
  auto comm2 = v2::comm(group, "tag-b");
  ```

  **No `group::as_comm()`**: `as_X()` conventionally implies a cheap zero-cost conversion;
  `MPI_Comm_create_from_group` is a collective operation and must not be named that way.
  Instead, `v2::comm` has a constructor taking a `group const&` plus an optional tag and
  info. When called with a temporary (`v2::comm(session.group_from_pset(...))`), the group
  is freed at the end of the full expression — exactly the short lifetime users want.
  Dependency direction is correct: `comm.hpp` → `group.hpp`.
  `session::comm_from_pset(pset, tag = "", info = MPI_INFO_NULL)` is a one-liner
  delegating to `v2::comm(group_from_pset(pset), tag, info)`.

- [ ] **`v2::psets` namespace** — port `psets::world` / `psets::self` constants from PR #772.

- [x] **`Group` at `mpi::experimental::` layer** (`include/mpi/group.hpp`) —
  `group_view` (non-owning) + `group` (owning, move-only).
  `MPI_Group` added to `builtin_handle` in `handle.hpp`.
  Core accessors implemented: `size()`, `rank()` (→ `optional`), `contains_self()`,
  `compare()`, `native()`. `group::from_native()` is public (needed by `comm.hpp`).
  `group` implicitly converts to `group_view`.
  Re-exported in `kamping::v2::` via `include/kamping/v2/group.hpp`.

  **Deferred** (not yet implemented):
  - `translate_rank` / `translate_ranks`
  - Set algebra: `intersection`, `difference`, `set_union`
  - Subgroup selection: `include`, `exclude`, `include_ranges`, `exclude_ranges`, `rank_range`

  **Dependency rule** upheld — `group.hpp` does not include `comm.hpp` or `session.hpp`.
  - `comm_view::group() → group` (`MPI_Comm_group`) — in `comm.hpp` ✓
  - `session::group_from_pset(pset) → group` — deferred with session work

## Handle types

- [x] **`mpi::experimental::status`** / **`status_view`** — done (`include/mpi/status.hpp`); re-exported from `kamping::v2::` via `include/kamping/v2/status.hpp`
- [x] **`mpi::experimental::comm_view`** — non-owning wrapper (`include/mpi/comm.hpp`); CRTP mixin `comm_accessors` with `.rank()`, `.size()`, `.native()`, `.group()`
- [x] **`mpi::experimental::request_view`** — non-owning wrapper (`include/mpi/request.hpp`); re-exported from `kamping::v2::` via `include/kamping/v2/request.hpp`; used in `iresult_base` do_wait/do_test
- [x] **`mpi::experimental::comm`** — owning RAII communicator (`include/mpi/comm.hpp`);
  move-only; `MPI_Comm_free` on destruction; `from_native()` static adoption factory;
  default constructor for use as MPI out-parameter; implicit conversion to `comm_view`.
  Re-exported in `kamping::v2::` via `include/kamping/v2/comm.hpp`.
- [x] **`dup` / `split` free functions** — `mpi::experimental::dup(Comm, NewComm&)` /
  `split(Comm, color, key, NewComm&)` templated on `convertible_to_mpi_handle` /
  `convertible_to_mpi_handle_ptr`; mirror C API in/out signature; return `void`.
  `comm::dup()` / `comm::split()` member methods delegate to these.
  `comm_view` intentionally has no `dup`/`split` — those are only on the owning `comm`.

## Request pool

Collects heterogeneous `iresult`s and waits for them in bulk via `MPI_Waitall`, which is
more efficient than waiting on each individually.

### Data model

```
request_pool
  pending_: vector<unique_ptr<iresult_base>>   ← type-erased ownership
  waited_:  bool                               ← guards get() before waitall()

ticket<Bufs...>
  idx_:   size_t           ← index into pool's pending_ vector
  pool_:  request_pool*    ← back-pointer; ticket.get() delegates to pool.get(*this)
```

`iresult_base` already exposes `mpi_native_handle_ptr() → MPI_Request*` — no interface
changes needed.

### `push(iresult<Bufs...>&&) → ticket<Bufs...>`

Moves the `iresult` into a `unique_ptr<iresult_base>` and appends to `pending_`. Returns a
typed `ticket` holding the index and a back-pointer to the pool.

### `waitall()`

```cpp
void waitall() {
    std::vector<MPI_Request> reqs;
    for (auto& r : pending_) reqs.push_back(*r->mpi_native_handle_ptr());
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    // Null out the stored requests so iresult_base destructors don't double-wait.
    for (auto& r : pending_) *r->mpi_native_handle_ptr() = MPI_REQUEST_NULL;
    waited_ = true;
}
```

### `ticket<Bufs...>::get()` / `pool.get(ticket<Bufs...>)`

```cpp
decltype(auto) get(ticket<Bufs...> const& t) {
    KAMPING_ASSERT(waited_, "call waitall() before get()");
    auto* r = static_cast<iresult<Bufs...>*>(pending_[t.idx_].get());
    return r->wait();  // MPI_Wait on MPI_REQUEST_NULL is a no-op; falls through to extract_buf()
}
```

No changes to `iresult` needed. The `static_cast` is safe because `push()` recorded the
concrete type in the ticket at call time.

### Destruction

No special destructor logic needed: if `waitall()` was not called, each `iresult_base`
destructor individually calls `MPI_Wait` on its own request (existing behavior). Correct
but less efficient than a single `MPI_Waitall` — the same trade-off as forgetting to call
`wait()` on a standalone `iresult`.

### `testsome()` (design open)

`MPI_Testsome` returns indices of completed requests. The pool collects non-null requests,
calls `MPI_Testsome`, and nulls out the completed ones. The open question is the return type:
tickets are heterogeneous (`ticket<SBuf>`, `ticket<RBuf>`, ...) so a plain vector doesn't
work. Candidates: return indices (`vector<size_t>`) with manual cast, callbacks registered
at push time, or restrict the pool to homogeneous buffer types. Needs a dedicated design
session before implementation.

## P2P

- [x] **`mpi::experimental::send`** / **`v2::send`** (all modes: standard, buffered, sync, ready)
- [x] **`mpi::experimental::recv`** / **`v2::recv`**
- [x] **`mpi::experimental::isend`** / **`v2::isend`**
- [x] **`mpi::experimental::irecv`** / **`v2::irecv`**
- [x] **`mpi::experimental::sendrecv`** / **`v2::sendrecv`**
- [x] **`mpi::experimental::isendrecv`** / **`v2::isendrecv`**
- [x] **`mpi::experimental::probe`** — bare `MPI_Probe` wrapper (core only)
- [x] **`mpi::experimental::mprobe`** — bare `MPI_Mprobe` wrapper (core only)
- [x] **`mpi::experimental::mrecv`** / **`mpi::experimental::imrecv`**
- [ ] **`probe_result` type** (`p2p/probe_result.hpp`)
  - Owns `MPI_Message` and `MPI_Status` from a matched probe
  - Accessors: `.source()`, `.tag()`, `.count<T>()`
  - `.mrecv(rbuf)` — blocking matched receive; resizes buffer from known count
  - `.imrecv(rbuf)` — non-blocking matched receive; returns `iresult<RBuf>`
- [ ] **`v2::mprobe`** / **`v2::improbe`** (`p2p/mprobe.hpp`)
  - `mprobe(source, tag, comm)` → `probe_result`
  - `improbe(source, tag, comm)` → `std::optional<probe_result>`
  - Update `infer(comm_op::recv, ...)` to use `probe_result` instead of raw `MPI_Message`

## Clean layer split + file restructuring

**Decision: views-free core.** The `core::` layer contains only the buffer contract (concepts,
accessor dispatch, `mpi_span`/`mpi_span_v`) and the bare MPI wrappers. All view machinery
(`view_interface`, `ref_view`, `owning_view`, `all()`, adaptor infrastructure, `with_*` views)
belongs exclusively to the language-bindings layer and must not appear in any header that
`core::` functions need to include.

The `core::` function signatures do **not** change — they remain concept-constrained templates
accepting anything satisfying `send_buffer` / `recv_buffer` / `send_buffer_v` / `recv_buffer_v`.
`mpi_span` and `mpi_span_v` are concrete minimal implementations of those concepts for callers
who prefer not to use the view pipeline.

**Core files move to `include/mpi/` now** (before the monorepo restructure). This establishes
the physical boundary immediately and makes the later monorepo step a trivial `git mv` with no
logic changes. The intermediate layout (actual filenames differ slightly from the plan):

```
include/
  mpi/                         ← core layer (views-free, self-contained) ✓ DONE
    collectives/
      allgather.hpp            (mpi::experimental:: namespace) ✓
      allgatherv.hpp           ✓
      alltoall.hpp             ✓
      alltoallv.hpp            ✓
      bcast.hpp                ✓
      barrier.hpp              ✓
    p2p/
      send.hpp                 ✓
      recv.hpp                 ✓
      isend.hpp                ✓
      irecv.hpp                ✓
      sendrecv.hpp             ✓
      isendrecv.hpp            ✓
      probe.hpp                ✓
      mprobe.hpp               ✓
      mrecv.hpp                ✓
      imrecv.hpp               ✓
    buffer.hpp                 (buffer concepts + accessor dispatch; was concepts.hpp+ranges.hpp) ✓
    mpi_span.hpp               (mpi_span / mpi_span_v) ← TODO Step 2
    handle.hpp                 (was native_handle.hpp) ✓
    error.hpp                  (was error_handling.hpp) ✓
  kamping/v2/                  ← ergonomics layer; #includes from <mpi/...>
    collectives/               (kamping::v2:: only; one-liners calling infer + mpi::experimental::)
    p2p/
    views/
    infer.hpp
    result.hpp
    iresult.hpp
    ...
  kamping/                     ← v1 untouched
```

### Namespace alignment

**Decision:** core layer moves to `mpi::experimental::`, ergonomics layer stays `kamping::v2::`.

| Was | Now (done) |
|---|---|
| `kamping::ranges::` (concepts, accessor dispatch) | `mpi::experimental::` ✓ |
| `kamping::core::` (bare MPI wrappers) | `mpi::experimental::` ✓ |
| `kamping::views::` (view adaptor factories) | `kamping::v2::views::` ✓ |
| `kamping::v2::` (high-level wrappers, infer) | `kamping::v2::` ✓ |

### Step-by-step tasks

**[x] Step 1 — Eliminate `ranges/` subdirectory; migrate namespaces; rename tags → sentinels**

  - `ranges/concepts.hpp` → `views/concepts.hpp` (namespace `kamping::v2::`) ✓
  - `ranges/ranges.hpp` content merged into `views/view_interface.hpp` ✓
  - `ranges/` directory deleted; top-level `ranges.hpp` deleted ✓
  - `tags.hpp` → `sentinels.hpp`; `resize_t`/`monotonic_t` moved into `views/view_interface.hpp` ✓
  - `kamping::ranges::` → `kamping::v2::` throughout all view and binding headers ✓
  - `kamping::views::` factories → `kamping::v2::views::` throughout ✓

- [x] **Step 2 — Add `mpi_span` / `mpi_span_v`** (`include/mpi/mpi_span.hpp`) ✓

  Concrete non-template structs satisfying the buffer concepts without any view machinery.
  `void*` covers both send (`void const*` is implicit) and recv:

  ```cpp
  struct mpi_span {
      void*          ptr;
      std::ptrdiff_t size;
      MPI_Datatype   type;

      void*          mpi_ptr()  noexcept       { return ptr; }
      std::ptrdiff_t mpi_count()  const noexcept { return size; }
      MPI_Datatype   mpi_type()  const noexcept { return type; }
  };

  struct mpi_span_v {
      void*        ptr;
      MPI_Datatype type;
      int const*   counts;      // per-rank element counts (length: comm_size)
      int const*   displs;      // per-rank displacements  (length: comm_size)
      int          comm_size;

      void*                mpi_ptr()   noexcept       { return ptr; }
      MPI_Datatype         mpi_type()   const noexcept { return type; }
      std::span<int const> mpi_counts() const noexcept { return {counts, static_cast<std::size_t>(comm_size)}; }
      std::span<int const> mpi_displs() const noexcept { return {displs, static_cast<std::size_t>(comm_size)}; }
  };
  ```

  - `mpi_span` satisfies `send_buffer` and `recv_buffer`
  - `mpi_span_v` satisfies `send_buffer_v` and `recv_buffer_v` (no scalar `mpi_count()`)

**[x] Step 3 — Split collectives + p2p files and move core halves to `include/mpi/`**

  Done. All collectives and p2p files are split; core halves live in `include/mpi/`.
  Infrastructure files placed as follows (final names differ from plan):
  - `ranges/` concepts + accessor dispatch → `include/mpi/buffer.hpp` (merged)
  - `native_handle.hpp` → `include/mpi/handle.hpp`
  - `error_handling.hpp` → `include/mpi/error.hpp`
  - CMakeLists updated to add `include/mpi` to include path.

**[x] Step 4 — Namespace rename**

  Done in the same pass as Step 3. All `include/mpi/` code is already in `mpi::experimental::`.
  All `include/kamping/v2/` references updated accordingly.

- [x] **Step 5 — Verify include discipline**

  Verified: no file under `include/mpi/` includes anything from `include/kamping/v2/views/`,
  `infer.hpp`, or result headers.

## Monorepo restructure

**Done.** The repo is split into four standalone components, each with its own `project()`,
`CMakeLists.txt`, and `FetchContent` bootstrap block.

Final layout:
```
mpi-core/          → kamping::mpi_core   (MPI + kassert + types, cxx_std_23)
kamping-v1/        → kamping::kamping    (original named-param API, cxx_std_17)
kamping-v2/        → kamping::v2         (concept-based buffer protocol, cxx_std_23)
ecosystem/cereal/  → kamping::ecosystem::cereal  (Cereal adapter for v2)
kamping-types/     → kamping::types      (shared type registry, submodule)
```

Each component is usable as `FetchContent SOURCE_SUBDIR` without pulling in the others.
All include paths and namespaces are unchanged from the user perspective.

CMake details:
- Every subproject has `cmake_minimum_required` + `project()` so `PROJECT_IS_TOP_LEVEL` works.
- `kamping_v2_warnings` INTERFACE target for warning flags; monorepo root appends `-Werror` when `KAMPING_WARNINGS_ARE_ERRORS=ON`.
- `mpi-core` sets a default `KAMPING_ASSERT_ASSERTION_LEVEL` (20 Debug / 10 Release) to avoid kassert's missing-level warning in v2-only builds.
- v2 tests and examples both link `kamping_v2_warnings`.
- v2 tests use standalone KaTestrophe v1.0.2 (FetchContent); v1's local `gtest-mpi-listener`/`mpi-gtest-main` targets renamed with `kamping-v1-` prefix to avoid conflicts.
- v1-only CMake options (`KAMPING_TESTS_DISCOVER`, `KAMPING_ENABLE_ULFM`, `KAMPING_EXCEPTION_MODE`, `KAMPING_ASSERTION_LEVEL`) live in `kamping-v1/CMakeLists.txt`.
- README updated with monorepo overview and per-component FetchContent snippets.

Include paths and namespace are settled: headers under `include/mpi/` (no `experimental/`
subdirectory), namespace `mpi::experimental::`. When standardized: one namespace rename,
no file moves. kamping-v2 stays `kamping::v2::` throughout.

## Reduction operation handling

Done. Implemented in `include/mpi/ops.hpp`. The final API names differ slightly from the plan:

| Planned | Implemented |
|---|---|
| `op_traits<Op, SBuf>` (single buffer) | `op_traits<Op, SBuf, RBuf>` (both buffers; allows asymmetric ops) |
| `mpi_op_for<Op, SBuf>` concept | `valid_op<Op, SBuf, RBuf>` concept |
| `native_op<SBuf>(op)` free function | `as_mpi_op(op, sbuf, rbuf)` free function |

Three-tier dispatch in `as_mpi_op`:
1. `op_traits<Op,SBuf,RBuf>` specialization — non-intrusive customization point
2. `convertible_to_mpi_handle<Op, MPI_Op>` — raw `MPI_Op` passthrough (already in `builtin_handle`)
3. `mpi_operation_traits<Op, range_value_t<Buf>>` — builtin functor inference from element type

`handle_traits` specializations for `ScopedOp` / `ScopedFunctorOp` / `ScopedCallbackOp` /
`ScopedDatatype` were intentionally not added: creating and lifetime-managing those objects
is the caller's responsibility at the core layer. Users pass `.get()` directly or wrap in
`op_traits`. v2 may add convenience wrappers if needed.

Inplace handling (branching on `ptr(sbuf) == MPI_IN_PLACE`) is implemented in `reduce` and
`allreduce`; the same pattern applies to `scan`/`exscan` when those are added.

## Sentinels (`include/kamping/v2/tags.hpp`)

Three zero-overhead sentinel buffer types. All implemented.

- [x] **`v2::inplace`** (`inplace_t`) — passes `MPI_IN_PLACE` / 0 / `MPI_DATATYPE_NULL`; satisfies
  `send_buffer` and `recv_buffer`. Used as the send argument for inplace collective overloads
  and available for explicit use in two-buffer forms.
- [x] **`v2::null_buf`** (`null_buf_t`) — passes `nullptr` / 0 / `MPI_DATATYPE_NULL`; satisfies
  `send_buffer` and `recv_buffer`. Used internally by non-root shorthand overloads; also
  exposed for users who want explicit uniform call sites across root and non-root.
- [x] **`v2::bottom`** (`bottom_t`) — passes `MPI_BOTTOM` only; exposes `mpi_data()` but **not**
  `mpi_size()` or `mpi_type()`. Not a `data_buffer` on its own — must be composed with
  `views::with_type` and `views::with_size` before use. The existing `view_interface`
  conditional forwarding already handles partial bases correctly; no view changes needed.

  ```cpp
  v2::send(v2::bottom | views::with_type(my_abs_type) | views::with_size(1), dest, comm);
  ```

## Type pool (`include/kamping/v2/type_pool.hpp`)

- [x] **`type_pool`** — move-only registry that owns committed `MPI_Datatype` handles for the
  lifetime of the pool. `register_type<T>()` commits on first call (idempotent); `find<T>()`
  is a const lookup returning `std::nullopt` if not yet registered. Builtin types always
  return immediately without storing anything.
- [x] **`views::with_pool`** — pipe adaptor; attaches the registered type from a `const` pool.
  Asserts the type was pre-registered; call `register_type<T>()` first.
- [x] **`views::with_auto_pool`** — pipe adaptor; takes a mutable pool ref and calls
  `register_type<T>()` lazily on first use.

## Derived-type view factories

Factories that create, commit, and own a derived `MPI_Datatype` inside a move-only closure.
The closure is assigned to a named variable; its lifetime bounds the type's validity. Multiple
MPI calls can reuse the same closure without re-committing. No `type_pool` needed — scope
manages the lifetime naturally.

```cpp
auto int_stride_2 = kamping::v2::make_strided_view<int>(2);
kamping::v2::send(sbuf | int_stride_2, dest, comm);
kamping::v2::send(sbuf2 | int_stride_2, dest, comm);  // reuses the same committed type
```

When applied to a range `r`, the resulting view exposes:
- `mpi_ptr()` → `std::ranges::data(r)` (original contiguous pointer)
- `mpi_type()` → the committed derived `MPI_Datatype` owned by the closure
- `mpi_count()` → element count adjusted for the derived type (e.g. `size / stride`)
- Optionally forwards `std::views::strided` / similar for iteration ergonomics

The closure is move-only (contains a `ScopedDatatype`); passing by lvalue ref in a pipe
borrows it safely via the existing `store_arg` `std::ref` path.

- [ ] **`make_strided_view<T>(stride)`** — commits `MPI_Type_vector(1, 1, stride, mpi_type<T>)`
  with resized extent so it tiles correctly; `mpi_count()` = `size / stride`.
- [ ] **`make_subarray_view<T>(...)`** — commits `MPI_Type_create_subarray` for multi-dimensional
  array sections; useful for halo exchange in structured-grid codes.
- [ ] **`make_struct_view<T>()`** — commits `MPI_Type_create_struct` from field offsets (via
  Boost.PFR reflection or manual specification); alternative to `byte_serialized` that
  respects MPI's struct type rules and avoids transmitting padding bytes.

## kamping-types: standard-library type specializations

Done. Shipped in PR #802 (`kamping-types-std-specializations` branch, cherry-picked to main).
Headers live under `kamping-types/include/kamping/types/std/` (not the `types/` root, to keep
opt-in extensions separate from core infrastructure):

- [x] **`kamping/types/std/utility.hpp`** — `std::pair<F, S>` via `struct_type` (safe)
- [x] **`kamping/types/std/tuple.hpp`** — `std::tuple<Ts...>` via `struct_type` (safe)
- [x] **`kamping/types/std/unsafe/utility.hpp`** — `std::pair<F, S>` via `byte_serialized`
- [x] **`kamping/types/std/unsafe/tuple.hpp`** — `std::tuple<Ts...>` via `byte_serialized`
- [x] **`kamping/types/std/unsafe/trivially_copyable.hpp`** — opt-in catch-all for any
  `std::is_trivially_copyable_v<T>` not already covered; excludes `std::pair`/`std::tuple`
  so it composes with the above headers.

## `recv_v<T>()` helper

- [x] **`v2::auto_recv_v<T, Cont = std::vector<T>>()`** / **`views::auto_recv_v`** — pipe adaptor and
  owned factory for the common variadic recv buffer idiom. The adaptor composes
  `auto_counts() | auto_displs() | resize_v`; the factory wraps an owned `Cont{}`:

  ```cpp
  v2::allgatherv(local_data, v2::auto_recv_v<int>(), comm);
  recv_buf | views::auto_recv_v   // lvalue borrow form
  ```

## Collectives

- [x] **`mpi::experimental::barrier`** / **`v2::barrier`**
- [x] **`mpi::experimental::bcast`** / **`v2::bcast`**
- [x] **`mpi::experimental::allgather`** / **`v2::allgather`**
- [x] **`mpi::experimental::allgatherv`** / **`v2::allgatherv`**
- [x] **`mpi::experimental::alltoall`** / **`v2::alltoall`**
- [x] **`mpi::experimental::alltoallv`** / **`v2::alltoallv`**
- [x] **`mpi::experimental::reduce`** / **`v2::reduce`** (core + v2 + infer); demonstrates inplace handling pattern for all reduction collectives
- [x] **`mpi::experimental::allreduce`** / **`v2::allreduce`** (core + v2 + infer); symmetric, no root
- [x] **Symmetric reduction**: `scan`, `exscan` — follow allreduce pattern exactly (inplace via sbuf=MPI_IN_PLACE; infer sets recv count from sbuf count when not inplace; all ranks symmetric so single `if constexpr (deferred_recv_buf<RBuf>)` works).
  - **`exscan` rank-0 caveat**: MPI leaves rank 0's recv buf undefined. v1 filled it with `values_on_rank_0` param or the op identity (for builtin ops). v2 does **not** add this: filling rank 0's buffer post-call is application logic, and op identity lookup requires type-level knowledge that doesn't belong in the buffer-protocol layer. Documented in the Doxygen; users fill it themselves if needed.
- [x] **`mpi::experimental::gather`** / **`v2::gather`** (core + v2 + infer); demonstrates asymmetric collective pattern
- [x] **`mpi::experimental::gatherv`** / **`v2::gatherv`** (core + v2 + infer); introduces `null_buf_v` (core) and `auto_null_recv_v()` (v2) for non-root deferred participation
- [x] **`mpi::experimental::scatter`** / **`v2::scatter`** (core + v2 + infer); two-buffer + non-root shorthand; inplace on root via MPI_IN_PLACE; infer uses MPI_Bcast of per-rank count
- [x] **`mpi::experimental::scatterv`** / **`v2::scatterv`** (core + v2 + infer); no shorthand (non-root passes `null_buf_v`); infer uses MPI_Scatter of sendcounts
- [ ] **Non-blocking** (defer): all `i*` variants — leverage iresult infrastructure from p2p
- Architecture proven: `core::` wraps MPI directly, `v2::` handles inference and result types

### Collective interface design

#### Inplace single-buffer overloads (symmetric collectives only)

Single-buffer = inplace, but **only** for collectives where all ranks have the same
relationship to the buffer. For asymmetric collectives (gather, scatter, reduce) the buffer
plays different roles on root vs non-root (different semantics, different sizes) — no
single-buffer form there.

| Collective | Single-buffer form | Notes |
|---|---|---|
| `allreduce(buf, op, comm)` | ✗ | Rejected: implicit inplace hurts readability; `allreduce(inplace, buf, op)` is already clean |
| `scan(buf, op, comm)` | ✗ | Same reasoning as allreduce |
| `exscan(buf, op, comm)` | ✗ | Same reasoning as allreduce |
| `allgather(buf, comm)` | ✓ | All ranks: local data already at `rank*n` slot, rest filled in |
| `gather` / `scatter` / `reduce` | ✗ | Buffer role differs by rank — always two-buffer |

The inplace allgather precondition (local data pre-placed at `rank * n`) is implicit and
documented but not enforced.

**Decision (allreduce/scan/exscan):** No implicit single-buffer inplace form. The asymmetry with
`reduce(sbuf, op, root)` (which means non-root send-only, not inplace) would make the rule
"single-buffer = inplace" inconsistent across collectives. Prefer the explicit
`allreduce(v2::inplace, buf, op)` which is already clean compared to the raw MPI C API.

#### Asymmetric collectives: gather / scatter / reduce

Three overload tiers — all ranks always call `infer()` regardless of which overload is used,
since `infer()` may itself be collective (e.g. `MPI_Bcast` inside scatter's infer to
distribute the per-rank count). Branching logic lives inside `infer()`.

```
gather(sbuf, root, comm)              // non-root shorthand — null_buf internally
gather(sbuf, rbuf, root, comm)        // full two-buffer; null_buf on non-root, real rbuf on root

scatter(rbuf, root, comm)             // non-root shorthand
scatter(sbuf, rbuf, root, comm)       // full two-buffer

reduce(sbuf, op, root, comm)          // non-root shorthand
reduce(sbuf, rbuf, op, root, comm)    // full two-buffer
```

`v2::null_buf` is exposed and usable in the two-buffer form for users who prefer uniform
call sites (root and non-root both call the 4-arg form, non-root passes `null_buf`).

**Rejected alternatives:**
- `std::optional<RBuf>` overload: does not eliminate the rank-check branch (moves it to
  construction site); deferred resize inside an optional is unspellable; extra `infer`
  overloads for every collective. Not worth the complexity.
- Single-buffer inplace for gather/scatter/reduce: buffer means recv on root, send on
  non-root — inconsistent role, inconsistent size. Rejected for asymmetric collectives.
