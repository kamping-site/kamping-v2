# Caching state on MPI objects

Libraries built on MPI often need state that belongs to a particular communicator (or window, or datatype) and
should live exactly as long as it does — for example:

- a counter window for the RSX protocol of `dstl::sparse_alltoall`, created on first use and reused by later calls,
- a private duplicate of a user's communicator, so the library's messages can never be matched by the user's,
- sub-communicators derived from a communicator (rows and columns of a process grid).

MPI provides *attributes* for this: values attached to an object under a key (a *keyval*), with a callback that runs
when the object is freed. `mpi::experimental::keyval` and `mpi::experimental::cache` (also available in
`kamping::v2`) wrap them as typed, RAII-managed caches. They differ only in **who owns the cached values**.

## Quick choice

| You have… | Use | Values are destroyed when… |
|---|---|---|
| an object that can own the state (a class member, a context, a local in `main`) | `cache<Kind, T>` | the MPI object is freed, the value is erased, or **the cache is destroyed** |
| nothing but the MPI object (stateless functions, a C API) | `keyval<Kind, T>` | the MPI object is freed or the value is erased |

Prefer `cache`: its lifetime is explicit, it frees everything it created, and it involves no global state, so it works
the same in the World and the Sessions model. Use `keyval` when there is no place to keep a cache.

## Object kinds

Both are parameterized by a *kind* tag that selects the attribute API:

| Kind | Objects | MPI functions |
|---|---|---|
| `attribute::comm` | communicators | `MPI_Comm_*_attr` |
| `attribute::win` | windows | `MPI_Win_*_attr` |
| `attribute::type` | datatypes | `MPI_Type_*_attr` |

The kind is a separate tag rather than the handle type because in some implementations (e.g. MPICH) `MPI_Comm`,
`MPI_Win` and `MPI_Datatype` are all `int`, so the handle type cannot tell them apart.

## `cache<Kind, T>`: the cache owns the values

```cpp
#include <mpi/cache.hpp>
#include <mpi/win.hpp>

namespace mx = mpi::experimental;

class exchanger {
    // One counter window per communicator this exchanger is used with.
    mx::cache<mx::attribute::comm, mx::win> _windows;

public:
    void exchange(MPI_Comm comm) {
        mx::win& w = _windows.get_or_create(comm, [&] { return mx::win_allocate<int>(1, comm).first; });
        w.fence(MPI_MODE_NOPRECEDE);
        // ...
    }
};
```

- `get_or_create(obj, make)` returns the value for `obj`, calling `make()` only the first time. `make` returns a `T` by
  value, which is constructed in place, so `T` need not be movable (like `win` here, which is move-only — or a type
  that is not even that).
- Each cache has its own key, so two caches on the same communicator never see each other's values.
- When the user frees a communicator, its value is destroyed and removed from the cache.
- When the cache is destroyed, it destroys all remaining values **in the order they were created**.

## `keyval<Kind, T>`: the object owns the value

```cpp
#include <mpi/attribute.hpp>

namespace mx = mpi::experimental;

// A stateless library function: nothing but `comm` survives between calls.
extern "C" void mylib_exchange(MPI_Comm comm) {
    static mx::keyval<mx::attribute::comm, mx::comm> private_comm;
    mx::comm& c = private_comm.get_or_create(comm, [&] { return mx::dup(comm); });
    // ... communicate on c, isolated from the user's traffic on comm
}
```

- Values live as long as their object: they are destroyed when the object is freed or the value is erased.
- Destroying the `keyval` only releases the key. MPI keeps it alive internally while attributes still use it, so the
  values stay attached — and are destroyed when their objects are freed — but can no longer be looked up.
- Values on objects that are never freed, most importantly `MPI_COMM_WORLD`, are never destroyed. For the example
  above this means a private duplicate of `MPI_COMM_WORLD` lives until the program ends.

## Rules for both

- **Collective creation and destruction.** If creating or destroying a `T` is collective (allocating or freeing a
  window, duplicating a communicator), then `get_or_create`, `erase` and freeing the object are collective too: all
  processes must call them for the same objects in the same order. For a `cache`, its destructor is then collective as
  well; destroying in creation order keeps it consistent across processes.
- **Finalization.** A `cache` must be destroyed before MPI is finalized. A `keyval` should be too; as a special case, a
  `keyval` in static storage (destroyed at program exit, after `MPI_Finalize`) skips freeing its key once MPI is
  finalized. In the Sessions model this cannot be detected, so destroy it before the last session is finalized.
- **Duplicates do not inherit.** `MPI_Comm_dup` and `MPI_Type_dup` create objects without cached values; they get
  their own on first use.
- **Constness follows ownership.** A `cache` owns its values, so `find` on a const cache returns `T const*`. A `keyval`
  is only a key, so `find` on a const keyval still returns `T*`.
- **No exceptions from destructors.** Values are destroyed inside MPI's callbacks, so `T`'s destructor must not throw.
- **Not thread-safe.** Do not call `get_or_create` / `erase` concurrently on the same cache or keyval.
