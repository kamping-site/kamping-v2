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

## Datatype handles

A `datatype` owns a committed derived datatype that you built with an MPI type constructor and frees it in its
destructor; `datatype::commit` takes ownership and commits. An `uncommitted_datatype` owns a datatype that is not
committed, such as a building block for another datatype; `std::move(u).commit()` turns it into a `datatype`. A
`datatype_view` refers to any datatype without owning it. `views::with_type` accepts any `buffer_datatype`, but not an
`uncommitted_datatype`; a temporary `datatype` passed to it is moved into the view.

```{doxygenclass} kamping::v2::datatype
:members:
```

```{doxygenclass} kamping::v2::uncommitted_datatype
:members:
```

```{doxygenclass} kamping::v2::datatype_view
:members:
```

```{doxygenconcept} kamping::v2::buffer_datatype
```

## Type constructors

The type constructors wrap the MPI ones. Each has two forms: the full form mirrors the MPI call and writes the new
datatype into its last argument, a raw `MPI_Datatype` or an empty `uncommitted_datatype`; the short form returns it
as an `uncommitted_datatype`. The new datatype is never committed. Base datatypes can be any datatype handle,
including an `uncommitted_datatype`. Array arguments are contiguous ranges, and their length is MPI's count.

```{doxygenoverloads} kamping::v2::type_contiguous
```

```{doxygenoverloads} kamping::v2::type_vector
```

```{doxygenoverloads} kamping::v2::type_create_hvector
```

```{doxygenoverloads} kamping::v2::type_create_struct
```

```{doxygenoverloads} kamping::v2::type_create_subarray
```

```{doxygenoverloads} kamping::v2::type_create_darray
```

```{doxygenoverloads} kamping::v2::type_create_resized
```

```{doxygenoverloads} kamping::v2::type_dup
```

`type_create_subarray` and `type_create_darray` take the storage order as an `ArrayOrder`: `ArrayOrder::c`
(`MPI_ORDER_C`, row-major) or `ArrayOrder::fortran` (`MPI_ORDER_FORTRAN`, column-major).

## Committing by hand

A datatype that is managed by hand, without an owning wrapper, is committed with `type_commit`.

```{doxygenoverloads} kamping::v2::type_commit
```

```{doxygenclass} kamping::v2::datatype_accessors
:members:
```
