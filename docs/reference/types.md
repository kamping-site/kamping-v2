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

A `datatype` owns a derived datatype that you built with an MPI type constructor: `datatype::from_native` commits it,
and the destructor frees it. A `datatype_view` refers to any datatype without owning it. `views::with_type` accepts
both, as well as a raw `MPI_Datatype`; a temporary `datatype` passed to it is moved into the view.

```{doxygenclass} kamping::v2::datatype
:members:
```

```{doxygenclass} kamping::v2::datatype_view
:members:
```

```{doxygenclass} kamping::v2::datatype_accessors
:members:
```
