# Derived datatypes

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

A datatype that is managed by hand, without an owning wrapper, is committed with `type_commit`.

```{doxygenoverloads} kamping::v2::type_commit
```

```{doxygenclass} kamping::v2::datatype_accessors
:members:
```
