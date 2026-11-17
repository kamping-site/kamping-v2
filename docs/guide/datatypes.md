# Derived datatypes

Elements of builtin types such as `int` or `double` are sent with the matching predefined MPI datatype;
`v2::builtin_datatype<T>()` returns it, and `v2::builtin_mpi_type<T>` checks that `T` has one. Enumerations are sent
with the datatype of their underlying type, and `std::vector<v2::unpacked_bool>` is a vector of booleans that MPI can
send, unlike the bit-packed `std::vector<bool>`. Your own types need a derived datatype that describes their layout,
and derived datatypes have to be committed before use and freed afterwards.

A `type_pool` takes care of this. It is a move-only registry that owns committed datatypes: it commits a datatype
the first time it is needed and frees all of them when it is destroyed.

```cpp
struct point { int id; double x, y; };

template <>
struct kamping::v2::datatype_traits<point> : kamping::v2::as_struct<point> {};

v2::type_pool pool;
std::vector<point> points = ...;
v2::send(points | views::with_auto_pool(pool), dest, comm);  // committed on first use
```

The layout of a type is described by specializing `kamping::v2::datatype_traits`, whose `datatype()` returns a new
`uncommitted_datatype`. Three bases cover the common cases:

- `as_struct<T>` describes a plain struct by its fields, found with Boost.PFR, without listing them. It skips the
  padding between the fields.
- `as_bytes<T>` describes a type as its `sizeof(T)` bytes, padding included. This is usually faster, since MPI
  implementations pack and unpack struct datatypes with gaps, but only correct if all processes use the same
  representation of `T`.
- `as_array<E, N>` describes a type as `N` consecutive `E`s, e.g. a `vec3` of three `float`s. C arrays and `std::array`
  get this datatype without a specialization.

`views::with_auto_pool(pool)` registers the datatype of the element type on first use, and `views::with_pool(pool)`
uses one you registered before with `pool.register_type<T>()`. A type is only sent with its derived datatype through
a pool; the library never looks up `datatype_traits` on its own.

`std::pair` and `std::tuple` get a datatype by including one of two headers: `kamping/v2/types/std/utility.hpp` and
`tuple.hpp` describe them by their members, the ones in `kamping/v2/types/std/unsafe/` as bytes.
`kamping/v2/types/std/unsafe/trivially_copyable.hpp` describes every other trivially copyable class as bytes. Include
the same variant in every translation unit of a program; mixing them violates the one-definition rule.

To avoid passing a pool around separately, `comm_view_with_pool` bundles a communicator with a pool and can be
passed wherever either is expected:

```cpp
v2::comm_view_with_pool env{world, pool};
v2::send(points | views::with_auto_pool(env), dest, env);
auto [_, all] = v2::allgatherv(points | views::with_auto_pool(env),
                               v2::auto_recv_v<point>(env), env);
```

For classes that `as_struct` cannot reflect, e.g. with constructors, private members or base classes, or to send
only some members, build the datatype with a `struct_builder`. Builtin members are deduced, other members get their
datatype explicitly, e.g. from their own `datatype_traits`:

```cpp
auto const particle_type = v2::struct_builder<particle>{}
                               .field(&particle::id)
                               .array_field(&particle::pos)          // std::array<double, 3>
                               .field(&particle::velocity, vec_type)
                               .build()
                               .commit();
v2::send(particles | views::with_type(particle_type), dest, comm);
```

## One-off datatypes

Layouts that are not a C++ type, such as a matrix column or a process's block of a distributed matrix, are built
with the type constructors (`type_vector`, `type_indexed`, `type_create_subarray`, `type_create_darray`, ...). They
return an `uncommitted_datatype`; `.commit()` turns it into a `datatype` that `views::with_type` accepts:

```cpp
using v2::distribution;
auto const dbl = v2::builtin_datatype<double>();

// The block of an n x n matrix that this process owns on a p_rows x p_cols grid.
auto const block = v2::type_create_darray(
    {{n, distribution::block, p_rows}, {n, distribution::block, p_cols}}, dbl, comm).commit();
v2::send(matrix | views::with_type(block) | views::with_count(1), 0, comm);
```

Where MPI takes parallel arrays, v2 takes one descriptor per block or dimension, e.g. as a braced list. The storage
order defaults to row-major:

```cpp
// blocks {displacement, length}
auto upper = v2::type_indexed({{0, 3}, {4, 2}, {8, 1}}, dbl);
// dimensions {size, start, subsize}
auto interior = v2::type_create_subarray({{rows, 1, rows - 2}, {cols, 1, cols - 2}}, dbl);
// one element per index, from any integer range
auto selected = v2::type_create_indexed_block(indices, dbl);
```

The forms that mirror MPI, with parallel arrays, are available as well.

An `uncommitted_datatype` also frees the datatype. A temporary one can be the base of another constructor; it is
freed right after the call, which MPI allows:

```cpp
// Consecutive columns of an n x n row-major matrix: one column, resized to start one element later.
auto const column = v2::type_create_resized(v2::type_vector(n, 1, n, dbl), 0, sizeof(double)).commit();
```

`views::with_type` borrows an lvalue `datatype` and takes ownership of a temporary. `datatype::commit(raw)` takes
ownership of a datatype built with a raw MPI call.

## Types without a datatype

Types that cannot be described by a datatype, such as a `std::map` or a tree, can be serialized with the Cereal
adapter; see {doc}`ecosystem`.
