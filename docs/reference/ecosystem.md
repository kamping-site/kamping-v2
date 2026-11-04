# Ecosystem adapters

Each adapter is a separate CMake component and target, enabled with the matching option (see
{doc}`../guide/getting-started`). See {doc}`../guide/ecosystem` for an introduction.

## Kokkos

Target `kamping::v2::kokkos`, option `KAMPING_V2_ENABLE_KOKKOS`.

```{doxygenview} kamping::v2::views::kokkos
```

```{doxygenoverloads} kamping::v2::views::auto_kokkos_view
```

```{doxygenclass} kamping::v2::kokkos_view
:members:
```

## Thrust

Target `kamping::v2::thrust`, option `KAMPING_V2_ENABLE_THRUST`.

```{doxygenstruct} mpi::experimental::buffer_traits< thrust::device_vector< T, Alloc > >
```

```{doxygenview} kamping::v2::views::thrust
```

## SYCL

Target `kamping::v2::sycl`, option `KAMPING_V2_ENABLE_SYCL`.

```{doxygenview} kamping::v2::views::sycl
```

```{doxygenclass} kamping::v2::sycl_view
:members:
```

## Cereal

Target `kamping::v2::cereal`, option `KAMPING_V2_ENABLE_CEREAL`.

```{doxygenview} kamping::v2::views::serialize
```

```{doxygenoverloads} kamping::v2::views::deserialize
```

```{doxygenclass} kamping::v2::serialization_view
:members:
```
