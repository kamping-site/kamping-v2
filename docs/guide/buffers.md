# Data buffers

MPI describes the data of every communication as a *data buffer*: a typed sequence of elements in memory. The C
interface spells a buffer out as three arguments — a pointer, an element count and an `MPI_Datatype` — and leaves
it to you to extract them from your containers. In KaMPIng-v2, a buffer is a single object: anything for which
these three properties can be obtained.

```cpp
std::vector<double> v = ...;
v2::send(v, dest, comm);                    // pointer, count and MPI_DOUBLE come from v
v2::send(v | std::views::take(n), dest, comm);  // the first n elements
```

## What makes a type a buffer

A type is a *data buffer* if the three accessor functions `ptr()`, `count()` and `type()` are defined for it. It
is a *send buffer* if its pointer converts to `void const*`, and a *receive buffer* if it converts to `void*` — so
a `std::vector<int> const&` can be sent from but not received into. These requirements are C++20 concepts
({cpp:concept}`kamping::v2::data_buffer`, {cpp:concept}`kamping::v2::send_buffer`,
{cpp:concept}`kamping::v2::recv_buffer`), checked at compile time.

Each accessor is looked up in three steps:

1. a specialization of `buffer_traits<T>`, for types you do not own;
2. a member function (`mpi_ptr()`, `mpi_count()`, `mpi_type()`), for types you control;
3. the standard range interface: `std::ranges::data()` for contiguous ranges, `std::ranges::size()` for sized
   ranges, and the builtin MPI datatype of the element type, e.g. `MPI_INT` for `int`.

Because of the third step, all contiguous standard containers and views of builtin types are buffers as they are:
`std::vector`, `std::array`, `std::span`, `std::string`, and views such as `std::views::take(n)` that keep the
range contiguous.

## Making your own types buffers

The three steps apply per accessor, so you only provide what cannot be derived. A `thrust::device_vector` is a
sized range of a builtin type, so `count()` and `type()` work already; only its pointer needs a specialization,
because `data()` returns a `thrust::device_ptr` rather than a raw pointer:

```cpp
template <typename T, typename Alloc>
struct mpi::experimental::buffer_traits<thrust::device_vector<T, Alloc>> {
    static T const* ptr(thrust::device_vector<T, Alloc> const& v) {
        return thrust::raw_pointer_cast(v.data());
    }
    static T* ptr(thrust::device_vector<T, Alloc>& v) {
        return thrust::raw_pointer_cast(v.data());
    }
};
```

The traits are specialized in `mpi::experimental`, where they are defined (see {doc}`../reference/customization`).
For a type you control, member functions do the same:

```cpp
struct particle_block {
    std::unique_ptr<double[]> data;
    std::size_t               n;

    double const* mpi_ptr() const { return data.get(); }
    double*       mpi_ptr() { return data.get(); }
    std::size_t   mpi_count() const { return n; }
    MPI_Datatype  mpi_type() const { return MPI_DOUBLE; }
};
```

The traits are not limited to ranges. A `type()` that returns a derived datatype describes non-contiguous memory,
and a buffer may even point to `MPI_BOTTOM` with absolute addresses in its datatype. For device memory you
allocate yourself, e.g. with `cudaMalloc`, no customization is needed at all: wrap it in a `std::span`.

## Variable-size buffers

Collectives with a different element count per process, such as `gatherv` or `alltoallv`, additionally need
per-rank counts and displacements. Since these belong to the buffer's memory layout, they are part of the buffer
rather than separate arguments: a *variable-size buffer* ({cpp:concept}`kamping::v2::data_buffer_v`) also provides
`counts()` and `displs()`, each a contiguous range of `int`. It does not need a scalar `count()`.

Standard containers carry no per-rank information, so you attach it with views:

```cpp
std::vector<double> data = ...;
std::vector<int> counts = ...;   // counts[i] elements go to rank i
auto send = data | views::with_counts(counts) | views::auto_displs;
v2::alltoallv(send, v2::auto_recv_v<double>(), comm);
```

`auto_displs` computes the displacements from the counts. See {doc}`views` for the views, and
{doc}`receive-buffers` for receive buffers that obtain their counts from the other processes.
