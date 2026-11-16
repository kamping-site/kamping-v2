# Derived datatypes

Elements of builtin types such as `int` or `double` are sent with the matching builtin MPI datatype. If you need
that datatype yourself, e.g. for a raw MPI call, `v2::builtin_type<T>::data_type()` returns it, and the concept
`v2::builtin_mpi_type<T>` checks that `T` has one. For your own
types, MPI needs a derived datatype that describes their memory layout. Derived datatypes have to be committed
before they are used and freed afterwards, which is easy to get wrong.

A `type_pool` takes care of this. It is a move-only registry that owns committed datatypes: it commits a datatype
the first time it is needed and frees all of them when it is destroyed.

```cpp
#include <kamping/types/struct_type.hpp>

struct point { int id; double x, y; };

template <>
struct kamping::types::mpi_type_traits<point> : kamping::types::struct_type<point> {};

v2::type_pool pool;
std::vector<point> points = ...;
v2::send(points | views::with_auto_pool(pool), dest, comm);  // committed on first use
```

The layout of a type is described by specializing `kamping::types::mpi_type_traits`. For plain structs,
`struct_type` derives it from the fields with Boost.PFR, without listing them; you can also describe a layout by
hand. `views::with_auto_pool(pool)` registers the datatype of the element type on first use, and
`views::with_pool(pool)` uses one you registered before with `pool.register_type<T>()`.

To avoid passing a pool around separately, `comm_view_with_pool` bundles a communicator with a pool and can be
passed wherever either is expected:

```cpp
v2::comm_view_with_pool env{world, pool};
v2::send(points | views::with_auto_pool(env), dest, env);
auto [_, all] = v2::allgatherv(points | views::with_auto_pool(env),
                               v2::auto_recv_v<point>(env), env);
```

## One-off datatypes

Some datatypes describe a layout rather than a C++ type, such as a column of a matrix or the block of a distributed
matrix that one process owns. Build them with the type constructors, which wrap the MPI ones: `type_contiguous`,
`type_vector`, `type_create_hvector`, `type_indexed`, `type_create_hindexed`, `type_create_indexed_block`,
`type_create_hindexed_block`, `type_create_struct`, `type_create_subarray`, `type_create_darray` and
`type_create_resized`. Each returns the new datatype as an `uncommitted_datatype`, which frees it at the end of its
lifetime. `std::move(u).commit()`, or `.commit()` on the returned temporary, commits it and turns it into a
`datatype`:

```cpp
std::array const gsizes{n, n};               // an n x n matrix in blocks
std::array const distribs{MPI_DISTRIBUTE_BLOCK, MPI_DISTRIBUTE_BLOCK};
std::array const dargs{MPI_DISTRIBUTE_DFLT_DARG, MPI_DISTRIBUTE_DFLT_DARG};
std::array const psizes{p_rows, p_cols};     // on a p_rows x p_cols grid
auto const block = v2::type_create_darray(comm.size(), comm.rank(), gsizes, distribs,
                                          dargs, psizes, v2::array_order::c, MPI_DOUBLE)
                       .commit();

v2::send(matrix | views::with_type(block) | views::with_count(1), 0, comm);
```

Array arguments such as `gsizes` are contiguous ranges (an `std::array`, a `std::vector`, a `std::span`, ...); their
length is the count MPI expects, e.g. the number of dimensions. The base datatype can be a builtin one like
`MPI_DOUBLE`, a `datatype` or an `uncommitted_datatype`. Each constructor also has a form that mirrors the MPI
call and writes the new datatype into a last argument, a raw `MPI_Datatype` or an empty `uncommitted_datatype`.

`views::with_type` borrows a `datatype` passed as an lvalue. A temporary is moved into the view instead, so
`views::with_type(std::move(u).commit())` frees the datatype once the view is gone. A `type_pool` can also take a
`datatype` with `pool.register_type<T>(std::move(dt))`. If you build a datatype with a raw MPI call,
`datatype::commit(raw)` takes ownership of it and commits it. `dt.dup()` copies a datatype and keeps its committed
state: the copy of a `datatype` is a `datatype`.

A `datatype` is always committed. Datatypes that only serve as building blocks for other datatypes need not be
committed, but still have to be freed. The `uncommitted_datatype` that the constructors return does both:
`views::with_type` rejects it at compile time, and a temporary one passed as the base datatype of another
constructor is freed right after the call, which MPI allows:

```cpp
// Consecutive columns of an n x n row-major matrix: one column, resized to start one element later.
auto const column_type =
    v2::type_create_resized(v2::type_vector(n, 1, n, MPI_DOUBLE), 0, sizeof(double)).commit();
```

## Types without a datatype

Types that cannot be described by a datatype, such as a `std::map` or a tree, can be serialized with the Cereal
adapter; see {doc}`ecosystem`.
