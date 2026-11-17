# Datatypes

## Builtin types

C++ types with a predefined MPI datatype, such as `int` (`MPI_INT`) or `double` (`MPI_DOUBLE`), and enumerations need
no datatype handling: contiguous ranges of them are buffers as they are. Predefined datatypes are never committed or
freed.

```{doxygenconcept} kamping::v2::builtin_mpi_type
```

```{doxygenoverloads} kamping::v2::builtin_datatype
```

```{doxygenclass} kamping::v2::unpacked_bool
:members:
```

## Derived datatypes

The datatype of a type is described by specializing `datatype_traits`, by hand or with one of the bases `as_struct`,
`as_array` and `as_bytes`. A `type_pool` owns committed `MPI_Datatype`s: it commits a datatype on first use and frees
it with the pool. The views `with_pool` and `with_auto_pool` attach a datatype from a pool to a buffer (see
{doc}`views`).

```{doxygenstruct} kamping::v2::datatype_traits
```

```{doxygenconcept} kamping::v2::has_datatype
```

```{doxygenstruct} kamping::v2::as_struct
:members:
```

```{doxygenstruct} kamping::v2::as_array
:members:
```

```{doxygenstruct} kamping::v2::as_bytes
:members:
```

```{doxygenstruct} kamping::v2::as_struct_tag
```

`std::pair` and `std::tuple` get a datatype by including `kamping/v2/types/std/utility.hpp` or `tuple.hpp`, which
describe them by their members, or their counterparts in `kamping/v2/types/std/unsafe/`, which describe them as
bytes. `kamping/v2/types/std/unsafe/trivially_copyable.hpp` describes every other trivially copyable class as bytes.

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

The free function `type_dup` does the same as the `dup()` members, and can also write into a raw handle.

```{doxygenoverloads} kamping::v2::type_dup
```

## Type constructors

The type constructors return an owned `uncommitted_datatype`. Each core form also has a variant that writes into a
last output argument. Where MPI takes parallel arrays, v2 adds forms that take one descriptor per block or dimension.

```{doxygenoverloads} kamping::v2::type_contiguous
```

```{doxygenoverloads} kamping::v2::type_vector
```

```{doxygenoverloads} kamping::v2::type_create_hvector
```

```{doxygenoverloads} kamping::v2::type_indexed
```

```{doxygenoverloads} kamping::v2::type_create_hindexed
```

```{doxygenoverloads} kamping::v2::type_create_indexed_block
```

```{doxygenoverloads} kamping::v2::type_create_hindexed_block
```

```{doxygenoverloads} kamping::v2::type_create_struct
```

```{doxygenoverloads} kamping::v2::type_create_subarray
```

```{doxygenoverloads} kamping::v2::type_create_darray
```

```{doxygenoverloads} kamping::v2::type_create_resized
```

```{doxygenenum} kamping::v2::array_order
```

### Descriptors

```{doxygenstruct} kamping::v2::block
:members:
```

```{doxygenstruct} kamping::v2::hblock
:members:
```

```{doxygenstruct} kamping::v2::struct_field
:members:
```

```{doxygenstruct} kamping::v2::subarray_dim
:members:
```

```{doxygenstruct} kamping::v2::darray_dim
:members:
```

```{doxygenenum} kamping::v2::distribution
```

### Struct builder

```{doxygenclass} kamping::v2::struct_builder
:members:
```

## Committing by hand

A datatype that is managed by hand, without an owning wrapper, is committed with `type_commit`.

```{doxygenoverloads} kamping::v2::type_commit
```

```{doxygenclass} kamping::v2::datatype_accessors
:members:
```
