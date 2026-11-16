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

The free functions `dup` and `split` do the same as the members, and can also write into an existing `comm`:

```{doxygenoverloads} kamping::v2::dup
```

```{doxygenoverloads} kamping::v2::split
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
