# Derived datatypes

Elements of builtin types such as `int` or `double` are sent with the matching builtin MPI datatype. For your own
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

## Types without a datatype

Types that cannot be described by a datatype, such as a `std::map` or a tree, can be serialized with the Cereal
adapter; see {doc}`ecosystem`.
