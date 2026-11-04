# MPI objects

MPI objects — communicators, groups, requests, datatypes and the others — are opaque handles in the C interface:
cheap to copy, but created and freed by explicit calls. A purely owning C++ wrapper does not fit MPI either:
`MPI_COMM_WORLD` and `MPI_COMM_SELF` exist from the start and must never be freed, and a library often uses objects
whose lifetime it does not control.

KaMPIng-v2 therefore provides both forms, shown here for communicators:

- `comm_view` does not own its communicator. It is constructed from any `MPI_Comm`, is cheap to copy, and never
  frees anything.
- `comm` owns its communicator: it is move-only and frees it on destruction. It is created by operations such as
  `dup()` and `split()`, or explicitly with `comm::from_native()`. `release()` gives up ownership and returns the
  handle; `free()` frees it explicitly, with errors reported as exceptions.

Both have the usual accessors, such as `rank()` and `size()`, and an owning object converts to its view. Groups,
info objects, sessions and windows follow the same pattern.

```cpp
v2::comm_view world{MPI_COMM_WORLD};
v2::comm      half = world.split(world.rank() % 2);
v2::send(data, 0, half);
```

## Your own types and native handles

You do not have to use these types. Every argument that stands for an MPI object accepts the native handle, the
KaMPIng-v2 types, and your own wrapper types:

```cpp
v2::send(data, dest, MPI_COMM_WORLD);   // native handle
v2::send(data, dest, world);            // comm_view
v2::send(data, dest, my_comm);          // your own type
```

For your own type, specialize `handle_traits` or give it an `mpi_handle()` member — the same three-step lookup as
for buffers (see {doc}`buffers`):

```cpp
template <>
struct mpi::experimental::handle_traits<my_comm> {
    static MPI_Comm handle(my_comm const& c) { return c.native(); }
};
```

Ranks and tags accept any integer type, and tags also scoped enumerations, which makes message kinds type-safe:

```cpp
enum class msg : int { data, done };
v2::send(buf, dest, msg::data, comm);
```

## Errors

KaMPIng-v2 follows the error handler of the communicator. With the default, `MPI_ERRORS_ARE_FATAL`, an error
terminates the program. With `MPI_ERRORS_RETURN`, errors are thrown as `mpi_error`:

```cpp
MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
try {
    v2::send(data, dest, MPI_COMM_WORLD);
} catch (v2::mpi_error const& e) {
    std::cerr << e.what() << '\n';
}
```

Destructors must not throw, so if freeing an owned object fails, the program terminates. Call `free()` explicitly
where you need to handle that error.
