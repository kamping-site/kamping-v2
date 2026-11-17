# MPI objects

MPI objects come in a non-owning form (`*_view`), which is cheap to copy and never frees the object, and an owning,
move-only form that frees it on destruction. Every function that takes an MPI object also accepts the native handle
(e.g. `MPI_Comm`) and your own wrapper types; see {doc}`customization` and {doc}`../guide/objects`.

With `MPI_ERRORS_RETURN` set on a communicator, errors are thrown as `mpi_error`; the default
`MPI_ERRORS_ARE_FATAL` terminates the program.

## Environment

An `environment` initializes and finalizes the World Model. Its static members query the World Model, also when MPI
was initialized by someone else. A session has a thread support level of its own (see
[Sessions](#sessions)).

```{doxygenclass} kamping::v2::environment
:members:
```

```{doxygenenum} kamping::v2::thread_level
```

## Communicators

The predefined communicators `MPI_COMM_WORLD` and `MPI_COMM_SELF` (see {ref}`world-and-sessions`):

```{doxygenoverloads} kamping::v2::comm_world
```

```{doxygenoverloads} kamping::v2::comm_self
```

```{doxygenclass} kamping::v2::comm_view
:members:
```

```{doxygenclass} kamping::v2::comm
:members:
```

The free functions `dup`, `split` and `split_type` do the same as the members, and can also write into an existing
`comm`. A process that passes `undefined` instead of a color or split type takes part in the split, but gets a null
`comm`:

```{doxygenoverloads} kamping::v2::dup
```

```{doxygenoverloads} kamping::v2::split
```

```{doxygenoverloads} kamping::v2::split_type
```

```{doxygenenum} kamping::v2::comm_type
```

```{doxygenvariable} kamping::v2::undefined
```

```{doxygenoverloads} kamping::v2::topo_test
```

```{doxygenenum} kamping::v2::topology
```

## Cartesian topologies

A Cartesian topology is a property of a communicator: `cart_create` returns an ordinary `comm`, and the functions
below take any communicator with a Cartesian topology. The communicator types also have members that delegate to
them (`grid.cart_coords()`, `grid.cart_position()`, `grid.cart_shift(d, k)`, `grid.cart_sub(d)`, ...).

```cpp
auto grid          = v2::cart_create(world, {{v2::auto_dim, true}, {v2::auto_dim, true}});  // balanced, periodic
auto [row, col]    = grid.cart_coords<2>();
auto [left, right] = grid.cart_shift(1, 1);              // positions, usable as ranks
v2::sendrecv(sbuf, right, rbuf, left, grid);
auto diag          = grid.cart_position().shift(0, 1).shift(1, 1);
v2::comm row_comm  = grid.cart_sub(1);                   // the processes of my row
```

```{doxygenoverloads} kamping::v2::cart_create
```

```{doxygenstruct} kamping::v2::cart_dim
:members:
```

```{doxygenvariable} kamping::v2::auto_dim
```

```{doxygenoverloads} kamping::v2::dims_create
```

```{doxygenclass} kamping::v2::cart_position
:members:
```

```{doxygenoverloads} kamping::v2::cart_shift
```

```{doxygenstruct} kamping::v2::cart_shift_result
:members:
```

```{doxygenoverloads} kamping::v2::cart_coords
```

```{doxygenoverloads} kamping::v2::cart_rank
```

```{doxygenoverloads} kamping::v2::cart_sub
```

```{doxygenoverloads} kamping::v2::cartdim
```

```{doxygenoverloads} kamping::v2::cart_get
```

```{doxygenoverloads} kamping::v2::cart_dims
```

```{doxygenconcept} kamping::v2::flag_array
```

## Groups

```{doxygenclass} kamping::v2::group_view
:members:
```

```{doxygenclass} kamping::v2::group
:members:
```

```{doxygenenum} kamping::v2::group_equality
```

## Status

```{doxygenclass} kamping::v2::status
:members:
```

```{doxygenclass} kamping::v2::status_view
:members:
```

```{doxygenclass} kamping::v2::const_status_view
:members:
```

```{doxygenclass} kamping::v2::status_vector
:members:
```

## Info objects

```{doxygenclass} kamping::v2::info_view
:members:
```

```{doxygenclass} kamping::v2::info
:members:
```

```{doxygenstruct} mpi::experimental::info_value_traits
```

## Sessions

```{doxygenclass} kamping::v2::session_view
:members:
```

```{doxygenclass} kamping::v2::session
:members:
```

The predefined process sets of MPI 4:

```{doxygenvariable} kamping::v2::psets::world
```

```{doxygenvariable} kamping::v2::psets::self
```

## Windows

```{doxygenclass} kamping::v2::win_view
:members:
```

```{doxygenclass} kamping::v2::win
:members:
```

```{doxygenoverloads} kamping::v2::win_allocate
```

```{doxygenoverloads} kamping::v2::fence
```

```{doxygenenum} kamping::v2::win_assert
```

## Caching state on MPI objects

See {doc}`../attribute_caching`.

```{doxygenclass} kamping::v2::cache
:members:
```

```{doxygenclass} kamping::v2::keyval
:members:
```

## Errors

```{doxygenstruct} kamping::v2::mpi_error
:members:
```
