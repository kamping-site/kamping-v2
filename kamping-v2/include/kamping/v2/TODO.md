# v2 TODO

## Cross-cutting: error handling (`std::expected`)

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
fires in debug builds when a count overflows `int`.

**No new CMake option.** MPI version is already detected by `find_package(MPI)`.

**When to implement:** in a single dedicated pass over all wrappers after the wrapper set
is complete. Adding guards wrapper-by-wrapper risks inconsistency and makes the diff
harder to review.

---

## Environment / Session

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
  `session::comm_from_pset(pset, tag = "", info = MPI_INFO_NULL)` is a one-liner
  delegating to `v2::comm(group_from_pset(pset), tag, info)`.

- [ ] **`v2::psets` namespace** — port `psets::world` / `psets::self` constants from PR #772.

### Group (deferred operations)

The following are not yet implemented in `mpi::experimental::group`:
- `translate_rank` / `translate_ranks`
- Set algebra: `intersection`, `difference`, `set_union`
- Subgroup selection: `include`, `exclude`, `include_ranges`, `exclude_ranges`, `rank_range`

---

## P2P

- [ ] **`probe_result` type** (`p2p/probe_result.hpp`)
  - Owns `MPI_Message` and `MPI_Status` from a matched probe
  - Accessors: `.source()`, `.tag()`, `.count<T>()`
  - `.mrecv(rbuf)` — blocking matched receive; resizes buffer from known count
  - `.imrecv(rbuf)` — non-blocking matched receive; returns `iresult<RBuf>`
- [ ] **`v2::mprobe`** / **`v2::improbe`** (`p2p/mprobe.hpp`)
  - `mprobe(source, tag, comm)` → `probe_result`
  - `improbe(source, tag, comm)` → `std::optional<probe_result>`
  - Update `infer(comm_op::recv, ...)` to use `probe_result` instead of raw `MPI_Message`

---

## Request pool: `testsome()`

`MPI_Testsome` returns indices of completed requests. The pool collects non-null requests,
calls `MPI_Testsome`, and nulls out the completed ones. The open question is the return type:
tickets are heterogeneous (`ticket<SBuf>`, `ticket<RBuf>`, ...) so a plain vector doesn't
work. Candidates: return indices (`vector<size_t>`) with manual cast, callbacks registered
at push time, or restrict the pool to homogeneous buffer types. Needs a dedicated design
session before implementation.

---

## Collectives

- [ ] **Non-blocking** (`i*` variants for all collectives) — leverage `iresult` infrastructure
  from p2p; architecture is proven, implementation is mechanical.

---

## Derived-type view factories

Factories that create, commit, and own a derived `MPI_Datatype` inside a move-only closure.
The closure is assigned to a named variable; its lifetime bounds the type's validity. Multiple
MPI calls can reuse the same closure without re-committing.

```cpp
auto int_stride_2 = kamping::v2::make_strided_view<int>(2);
kamping::v2::send(sbuf | int_stride_2, dest, comm);
kamping::v2::send(sbuf2 | int_stride_2, dest, comm);  // reuses the same committed type
```

When applied to a range `r`, the resulting view exposes:
- `mpi_ptr()` → `std::ranges::data(r)` (original contiguous pointer)
- `mpi_type()` → the committed derived `MPI_Datatype` owned by the closure
- `mpi_count()` → element count adjusted for the derived type (e.g. `size / stride`)

The closure is move-only (contains a `ScopedDatatype`); passing by lvalue ref in a pipe
borrows it safely via the existing `store_arg` `std::ref` path.

- [ ] **`make_strided_view<T>(stride)`** — commits `MPI_Type_vector(1, 1, stride, mpi_type<T>)`
  with resized extent so it tiles correctly; `mpi_count()` = `size / stride`.
- [ ] **`make_subarray_view<T>(...)`** — commits `MPI_Type_create_subarray` for multi-dimensional
  array sections; useful for halo exchange in structured-grid codes.
- [ ] **`make_struct_view<T>()`** — commits `MPI_Type_create_struct` from field offsets (via
  Boost.PFR reflection or manual specification); alternative to `byte_serialized` that
  respects MPI's struct type rules and avoids transmitting padding bytes.

---

## CMake / Build system

### Target naming convention

| Target | Alias | Notes |
|--------|-------|-------|
| `mpi_core` | `mpi::core` | Rename from current `kamping::mpi_core`; anticipates future split into its own repo |
| `kamping_v2` | `kamping::v2` | Unchanged |
| `kamping_v2_warnings` | `kamping::v2::warnings` | Unchanged |
| `kamping_ecosystem_<name>` | `kamping::ecosystem::<name>` | Uniform `::` separator; PR #7 has `kamping::ecosystem_kokkos` which must be fixed |

### Standalone FetchContent usage

Each sublibrary is independently consumable. Downstream projects point `FetchContent_Declare` at
the monorepo with a `SOURCE_SUBDIR` argument:

```cmake
FetchContent_Declare(kamping_v2
    GIT_REPOSITORY https://github.com/kamping-site/kamping-v2
    GIT_TAG <tag>
    SOURCE_SUBDIR kamping-v2
)
FetchContent_MakeAvailable(kamping_v2)
target_link_libraries(myapp PRIVATE kamping::v2)
```

`SOURCE_SUBDIR ecosystem/thrust`, `SOURCE_SUBDIR ecosystem/cereal`, etc. work the same way.
The whole monorepo is downloaded once; subsequent `FetchContent_Declare` calls for other
sublibraries reuse the cached download.

### Bootstrapping pattern

When a sublibrary's `CMakeLists.txt` is the CMake entry point (i.e. the monorepo root was not
processed), it must bootstrap its own dependencies. The pattern used in every sublibrary:

```cmake
if (NOT TARGET <required-target>)
    set(_local "${CMAKE_CURRENT_SOURCE_DIR}/<relative-path-to-dep>")
    if (NOT EXISTS "${_local}/CMakeLists.txt")
        message(FATAL_ERROR
            "<component>: <required-target> not found and the sibling directory "
            "'${_local}' does not exist. Provide <required-target> before including "
            "this component, or build from the monorepo root.")
    endif ()
    FetchContent_Declare(<dep> SOURCE_DIR "${_local}")
    FetchContent_MakeAvailable(<dep>)
endif ()
```

This works because `FetchContent` with `SOURCE_SUBDIR kamping-v2` downloads the whole monorepo,
so `../mpi-core` resolves correctly inside the fetched tree. When `mpi-core` eventually splits
into its own repo, only the `FATAL_ERROR` branch changes: replace `SOURCE_DIR` with
`GIT_REPOSITORY`/`GIT_TAG`.

**No shared bootstrap function.** The pattern is ~8 lines and appears in 4–5 places. Factoring
it into a shared `.cmake` module requires a reliably-findable path across all consumption modes,
which is more fragile than the repetition.

### Ecosystem adapter dependencies

GPU adapters (thrust, kokkos, sycl) bundle both buffer traits (`mpi::core` only) and view
adaptors (need `kamping::v2`). Since the view adaptors include `kamping/v2/views/` headers
directly, the CMake target must link `kamping::v2` — there is no meaningful split into a
`::core` sub-target. Users who want only buffer traits still pull in `kamping::v2` transitively,
but as a header-only target this has no compile cost unless those headers are included.

Cereal likewise depends on `kamping::v2` (serialize/deserialize views are v2-specific).

### Known issues to fix

- [x] Rename `kamping::mpi_core` alias to `mpi::core` in `mpi-core/CMakeLists.txt`; update all
  references in `kamping-v2/CMakeLists.txt` and ecosystem files.
- [x] `mpi-core/CMakeLists.txt` has no bootstrapping block. If used as a standalone
  `SOURCE_SUBDIR mpi-core` entry point, `MPI::MPI_CXX`, `kamping::kassert`, and
  `kamping::types` are undefined. Add the same bootstrapping pattern.
- [x] `kamping-v2/CMakeLists.txt` bootstrap uses `FetchContent_Declare(mpi_core SOURCE_DIR ...)`
  without checking existence first. Add the `FATAL_ERROR` guard matching the cereal/thrust pattern.
- [ ] `kamping_types` is pinned to `GIT_TAG main` in the root and in the v2 bootstrap — unstable.
  Pin to a release tag once kamping-types cuts one.
- [x] `KAMPING_ENABLE_SERIALIZATION` defaults `ON` in `ecosystem/CMakeLists.txt`. This silently
  fetches Cereal for every consumer. Change default to `OFF` to match `KAMPING_ENABLE_THRUST`.
- [x] PR #7 (Kokkos): fix alias to `kamping::ecosystem::kokkos`, add `option()` declaration with
  default `OFF`, pin `KokkosComm` `GIT_TAG` away from `develop`.
