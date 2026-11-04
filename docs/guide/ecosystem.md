# GPUs and the HPC ecosystem

Many MPI implementations can communicate directly between GPU memory regions, without copying through the host.
The C interface still needs a raw pointer, a count and a datatype, so GPU containers have to be taken apart by hand.
With KaMPIng-v2, adapting a container to the buffer concepts once is enough to use it with every operation.

The simplest case needs no adapter at all: device memory allocated with `cudaMalloc` or `sycl::malloc_device` can
be wrapped in a `std::span` and passed to any operation.

Each adapter is a separate CMake component, enabled with `KAMPING_V2_ENABLE_<ADAPTER>` (see
{doc}`getting-started`), and lives in `kamping/v2/ecosystem/`. The {doc}`reference <../reference/ecosystem>` lists
the details.

## Kokkos

`views::kokkos` makes a `Kokkos::View` a buffer. Contiguous views are passed to MPI directly, after fencing the
execution space; non-contiguous views are packed into a contiguous buffer when first accessed and unpacked after
receiving, following KokkosComm. Because the result is a buffer, it works with every operation — not only those
that have been implemented for Kokkos — and composes with the other views:

```cpp
Kokkos::View<int*> sv("send", n);
v2::send(sv | views::kokkos, dest, comm);

Kokkos::View<int*> rv("recv", 0);
v2::recv(rv | views::kokkos | views::resize, source, comm);  // resized to fit
```

With `KAMPING_V2_ENABLE_KOKKOS_COMM`, the adapter uses KokkosComm for packing.

## Thrust

A `thrust::device_vector` is a sized range of a builtin type, but its `data()` returns a `thrust::device_ptr`
instead of a raw pointer. Including `<kamping/v2/ecosystem/thrust_buffer_traits.hpp>` adds the `buffer_traits`
specialization that extracts the raw pointer, which makes device vectors buffers directly:

```cpp
thrust::device_vector<double> d(n);
v2::send(d, dest, comm);                         // with thrust_buffer_traits.hpp
v2::recv(d | views::thrust | views::resize, source, comm);
```

`views::thrust` does the same as a view, e.g. for vectors you want to resize.

## SYCL

Data in a `sycl::buffer` is accessed through accessors, and the pointer to device memory is only available inside
a host task, through its `sycl::interop_handle`. `views::sycl(ih)` follows this model: inside a host task, it makes
an accessor a buffer for the duration of the task:

```cpp
sycl::buffer<int> buf(...);
q.submit([&](sycl::handler& h) {
    sycl::accessor acc{buf, h};
    h.host_task([=](sycl::interop_handle ih) {
        v2::send(acc | views::sycl(ih), dest, comm);
    });
});
```

A `sycl::host_accessor` needs no interop handle: `hacc | views::sycl`.

## Cereal serialization

`views::serialize` serializes any object that Cereal can serialize, and `views::deserialize<T>()` creates a
receive buffer that deserializes into a new `T`. The receive buffer is deferred, so its size is obtained as for
`views::resize`:

```cpp
std::map<std::string, int> m = ...;
v2::send(m | views::serialize, dest, comm);

auto received = v2::recv(views::deserialize<std::map<std::string, int>>(), source, comm);
std::map<std::string, int> result = *received;
```
