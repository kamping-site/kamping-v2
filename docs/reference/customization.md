# Customization points

KaMPIng-v2 finds the MPI properties of a buffer, and the native handle of an MPI object, through a small set of
accessor functions. Each one is looked up in three steps: a traits specialization, for types you don't own; a member
function, for types you control; and a fallback for standard ranges and native MPI handles. See
{doc}`../guide/buffers` and {doc}`../guide/objects`.

The traits and concepts are defined in mpi-core, in `mpi::experimental`, and re-exported in `kamping::v2`.
Specialize the traits in `mpi::experimental`.

## Buffers

```{doxygenstruct} mpi::experimental::buffer_traits
```

```{doxygenoverloads} kamping::v2::count
```

```{doxygenoverloads} kamping::v2::ptr
```

```{doxygenoverloads} kamping::v2::type
```

```{doxygenoverloads} kamping::v2::counts
```

```{doxygenoverloads} kamping::v2::displs
```

```{doxygenoverloads} kamping::v2::resize_for_receive
```

```{doxygenvariable} kamping::v2::enable_borrowed_buffer
```

```{doxygenconcept} kamping::v2::borrowed_buffer
```

## MPI objects

```{doxygenstruct} mpi::experimental::handle_traits
```

```{doxygenoverloads} kamping::v2::handle
```

```{doxygenoverloads} kamping::v2::handle_ptr
```

```{doxygenconcept} kamping::v2::convertible_to_mpi_handle
```

```{doxygenconcept} kamping::v2::convertible_to_mpi_handle_ptr
```

### Arrays of handles

```{doxygenstruct} mpi::experimental::handle_range_traits
```

```{doxygenoverloads} kamping::v2::handle_range
```

```{doxygenconcept} kamping::v2::convertible_to_mpi_handle_range
```

### Ranks and tags

```{doxygenstruct} mpi::experimental::rank_traits
```

```{doxygenstruct} mpi::experimental::tag_traits
```

```{doxygenoverloads} kamping::v2::to_rank
```

```{doxygenoverloads} kamping::v2::to_tag
```

```{doxygenconcept} kamping::v2::rank
```

```{doxygenconcept} kamping::v2::tag
```

## Writing your own views

A view wraps a buffer, forwards the accessors it does not override and overrides the others as members. Derive from
`view_interface` to get the forwarding, and use `adaptor` or `adaptor_closure` to make it pipeable.

```{doxygenstruct} kamping::v2::view_interface
:members:
```

```{doxygenconcept} kamping::v2::view
```

```{doxygenoverloads} kamping::v2::all
```

```{doxygenclass} kamping::v2::ref_view
```

```{doxygenclass} kamping::v2::owning_view
```

```{doxygenstruct} kamping::v2::adaptor
```

```{doxygenstruct} kamping::v2::adaptor_closure
```

```{doxygenstruct} kamping::v2::adaptor_factory
```
