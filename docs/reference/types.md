# Datatypes

## Builtin types

C++ types with a predefined MPI datatype, such as `int` (`MPI_INT`) or `double` (`MPI_DOUBLE`), need no datatype
handling: contiguous ranges of them are buffers as they are. The mapping comes from kamping-types and is re-exported in
`kamping::v2`. `builtin_type<T>::data_type()` returns the predefined datatype of `T`, e.g. for raw MPI calls, and
`is_builtin_type_v<T>` tells whether there is one. Predefined datatypes are never committed or freed.

```{doxygenconcept} kamping::v2::builtin_mpi_type
```

## Derived datatypes

A `type_pool` owns committed `MPI_Datatype`s: it commits a datatype on first use and frees it with the pool. The
views `with_pool` and `with_auto_pool` attach a datatype from a pool to a buffer (see {doc}`views`). The datatype of
a type is described by specializing `kamping::types::mpi_type_traits`, by hand or for plain structs with
`kamping::types::struct_type`, which derives it from the fields with Boost.PFR.

```{doxygenclass} kamping::v2::type_pool
:members:
```

```{doxygenclass} kamping::v2::comm_view_with_pool
:members:
```

```{doxygenconcept} kamping::v2::has_pool
```
