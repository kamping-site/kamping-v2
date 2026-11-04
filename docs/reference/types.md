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
