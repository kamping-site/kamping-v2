# Design: layout (derived) datatype builders

Scope extension beyond [`value_type.md`](value_type.md). `value_type` and
`with_type` attach *metadata* to a contiguous buffer. This note covers the other
half of MPI's type system: **non-contiguous layout types** — strided, indexed,
subarray, resized — that describe scattered memory so it can be transferred
zero-copy.

The motivating observation (see discussion): these are **not** "more lazy
views". Two properties force a different shape than `value_type`:

1. A layout type **composes over an `oldtype`** (`MPI_Type_vector`, … all take a
   base type), so it is an algebra, not a flat annotation — and the base type
   may itself be a custom type.
2. Building one requires `MPI_Type_commit`, which is **expensive** and must be
   paid **once and reused**, never per pipe in a loop.

So the abstraction is a **builder + cache**, with `with_type` as the thin,
zero-cost attach over a pre-built committed handle. Lazy `|` that commits inline
is only a throwaway convenience.

---

## Core principle: commit once, reuse many

`MPI_Type_commit` may build an optimized internal representation; committing +
freeing on every operation is catastrophic in the loops where layout types are
actually used (halo exchange, repeated column sends). The design therefore makes
the commit point **explicit and amortizable**:

```cpp
// build + commit ONCE
auto col_t = make_strided_type(type(matrix), /*count=*/rows, /*blocklen=*/1, /*stride=*/cols);

// attach only (no commit) in the hot loop
for (int step = 0; step < n; ++step)
    kamping::v2::send(matrix | views::with_type(col_t), dst, comm);
```

`make_*_type` returns a `kamping::types::ScopedDatatype` (RAII: commits on
construction, frees on destruction). The caller owns the lifetime, exactly like
the `type_pool` / non-blocking contract in `value_type.md`.

---

## Composition algebra

Every builder takes an `oldtype` as its base. The default is deduced from the
buffer (`type(buf)`), but it may be any committed handle — including a custom
struct or `byte_serialized` type, or another layout type:

```cpp
auto cell_t   = pool.register_type<Cell>(struct_type<Cell>::data_type()); // custom base
auto column_t = make_strided_type(cell_t, rows, 1, cols);                 // strided-of-struct
auto plane_t  = make_strided_type(column_t, depth, 1, rows * cols);       // strided-of-strided
```

**Extent is load-bearing.** `MPI_Type_vector`'s stride is in multiples of the
oldtype's *extent*; `MPI_Type_indexed`'s displacements likewise. For a padded
struct, or any composed type, the extent is not `sizeof` — so striding/indexing
over a custom base only does the right thing when the base extent is correct.
This is why `resized` is part of the family, not an afterthought:

```cpp
auto packed_t = make_resized_type(struct_type<Cell>::data_type(), /*lb=*/0, /*extent=*/sizeof(Cell));
```

The byte-stride builders (`hvector`, `hindexed`) sidestep extent entirely by
taking strides/displacements in bytes — offer both, defaulting to the
element-count form and documenting the extent dependency.

---

## Builder catalog

| Builder | MPI call | Stride/displ unit |
|---|---|---|
| `make_strided_type(old, count, blocklen, stride)`        | `MPI_Type_vector`               | oldtype extents |
| `make_strided_type_b(old, count, blocklen, byte_stride)` | `MPI_Type_create_hvector`       | bytes |
| `make_indexed_type(old, blocklens, displs)`              | `MPI_Type_indexed`              | oldtype extents |
| `make_indexed_type_b(old, blocklens, byte_displs)`       | `MPI_Type_create_hindexed`      | bytes |
| `make_indexed_block_type(old, blocklen, displs)`         | `MPI_Type_create_indexed_block` | oldtype extents |
| `make_subarray_type(old, sizes, subsizes, starts[, order])` | `MPI_Type_create_subarray`   | — |
| `make_resized_type(old, lb, extent)`                     | `MPI_Type_create_resized`       | bytes |

All return `ScopedDatatype`. Intermediate (base) types need not be re-committed
— MPI only requires the type actually passed to a communication call to be
committed — but in practice the base usually comes from a pool already committed
for its own use, and that is harmless.

### Large counts (follow the `feat/large-count` convention)

PR #38 establishes a **purely compile-time, type-driven** rule (per the author:
"we should not dispatch this at runtime — if we have MPI-4 and the buffer has
large count, we unconditionally call the `_c` variant"): when the
count/displacement *types* are wide (`has_large_count` =
`sizeof(decltype(count(t))) > sizeof(int)`; variadic `_c` concepts require
`MPI_Count[]` counts / `MPI_Aint[]` displs) and `MPI_VERSION >= 4`, the `_c` API
is called unconditionally. There is **no** runtime `count > INT_MAX` branch; the
narrow path keeps only a defensive `count <= std::numeric_limits<int>::max()`
assertion inside the int-typed call itself, not as a dispatch decision.

The builders mirror this — dispatch on the parameter type, no runtime check:

```cpp
template <std::integral Count, std::integral Stride>
ScopedDatatype make_strided_type(MPI_Datatype old, Count count, Count blocklen, Stride stride) {
    MPI_Datatype t;
#if MPI_VERSION >= 4
    if constexpr (sizeof(Count) > sizeof(int) || sizeof(Stride) > sizeof(int))
        MPI_Type_vector_c(count, blocklen, stride, old, &t);   // MPI_Count params, unconditional
    else
#endif
        MPI_Type_vector(static_cast<int>(count), static_cast<int>(blocklen), static_cast<int>(stride), old, &t);
    return ScopedDatatype{t};   // commits
}
```

`make_indexed_type` / `make_subarray_type` take `MPI_Count[]` / `MPI_Aint[]`
(byte-displacement form) under MPI-4, falling back to `int[]`. `make_resized_type`
already uses `MPI_Aint` for `lb`/`extent` in both forms. The `layout_key` cache
records the chosen count type so a wide-count and a narrow-count layout with the
same numeric parameters do not collide. (PR #38 also floats a feature flag to let
users override the dispatch; if it lands, the builders should honour the same
flag.)

---

## `(ptr, count, type)` restructuring

A layout view is **not** metadata-only like `with_type`: it folds the access
pattern into the datatype and changes the buffer's shape. After
`matrix | strided(cols, 1)`:

- `mpi_ptr()` — unchanged (base of the first element).
- `mpi_type()` — the composite vector type (carries the layout).
- `mpi_count()` — collapses to `1` (the whole strided pattern is one message),
  not the element count.

So a layout view restructures all three accessors together, whereas `with_type`
/ `with_value_type` only touch `type` / the value channel. Builders make this
explicit (the caller passes the block count); the pipe sugar below must compute
it from the buffer size and set `mpi_count() == 1`.

---

## Parametric type cache (extending `type_pool`)

`type_pool` keys on `type_index(typeid(T))` — a compile-time C++ type. A layout
type is identified by **runtime parameters** `(oldtype, kind, count, blocklen,
stride, …)`, so caching needs a structural key, not a `type_index`.

Add a sibling registry (or a second map inside `type_pool`):

```cpp
struct layout_key {
    MPI_Datatype oldtype;        // base handle
    layout_kind  kind;           // vector / hvector / indexed / subarray / resized
    std::vector<long> params;    // counts, blocklens, strides/displs, sizes…  (flattened)
    bool operator==(layout_key const&) const = default;
};
// std::unordered_map<layout_key, ScopedDatatype, layout_key_hash>

MPI_Datatype type_pool::get_or_build(layout_key key, auto&& build_fn);
```

`build_fn` constructs + commits only on a cache miss; hits return the stored
handle. The pool owns every committed layout type for its lifetime, and because
the base `oldtype` is (typically) also pool-owned, **lifetime composes**: the
strided type stays valid as long as the pool — which also keeps its base alive —
is alive.

> Caveat: `MPI_Datatype` as a key assumes handle stability and equality. Two
> distinct-but-identical base types produce distinct cache entries (acceptable —
> just a missed dedup). Freeing a base while a cached layout references it is UB;
> the single-pool-ownership model avoids it.

---

## Pipe sugar (optional, on top of builders)

Two pipe forms, mirroring `with_pool` / throwaway:

```cpp
// cached: commit once across iterations, keyed in the pool
matrix | views::strided(cols, 1) | views::with_auto_pool(p);

// throwaway: builds + commits inline, frees when the view dies — one-shot only
matrix | views::strided(cols, 1);   // documented footgun in a loop
```

The cached form is the recommended ergonomic path; the throwaway form exists for
one-off calls and must carry a clear "commits per use" warning. Both compute the
block count from the buffer and set `mpi_count() == 1` per the restructuring
above.

`strided`/`indexed`/`subarray`/`resized` adaptor factories live in
`kamping::v2::views`, each forwarding to the corresponding `make_*_type` builder
(throwaway) or `pool.get_or_build` (cached).

---

## Layering

- **kamping-types** — new construction support: `make_strided_type` &
  friends, alongside the existing `contiguous_type` / `struct_type`. This is
  where the `MPI_Type_*` calls and `ScopedDatatype` wrapping live (no view
  machinery, mirrors the core/types split).
- **kamping-v2** — the `layout_key` cache extension to `type_pool`, the
  `views::strided`/… adaptors, and the `(ptr, count, type)` restructuring view.
- **dstl** — consumers (e.g. a column/halo exchange) build once and attach via
  `with_type`. `redistribute.hpp`'s manual `MPI_Type_get_extent` math is the
  kind of code a `resized`/extent builder would replace.

---

## Open questions

- **Block count source.** Does `views::strided(stride, blocklen)` derive the
  block count from the buffer size, or take it explicitly? Deriving is ergonomic
  but assumes the buffer length is an exact multiple; explicit is safer. Likely
  offer both (the builder takes it; the pipe derives + asserts divisibility).
- **Cache key normalization.** Should `(vector, count, blocklen, stride)` and an
  equivalent `hvector` with `stride*extent` dedup to one entry? Probably not
  worth it — keep keys literal.
- **Big counts** — resolved: builders follow the type-driven `_c` dispatch from
  PR #38 (see "Large counts" above), not an open decision.
- **Interaction with `value_type`.** A layout type built over a structured
  (`value_destination_pair`/nested) buffer is out of scope — layout types apply
  to flat buffers with a single base type. Structured buffers go through
  `flatten_v` first, then a layout type could wrap the flattened result.
