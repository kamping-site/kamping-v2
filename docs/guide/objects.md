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
v2::comm_view world = v2::comm_world();
v2::comm      half = world.split(world.rank() % 2);
v2::send(data, 0, half);
```

(world-and-sessions)=
## The world model and the sessions model

MPI offers two ways to get a first communicator.

In the **world model**, MPI is initialized once per process, by {cpp:class}`kamping::v2::environment` or any other
call to `MPI_Init`, and provides two predefined communicators: `MPI_COMM_WORLD` with all processes started together
and `MPI_COMM_SELF` with only the calling one. {cpp:any}`comm_world() <kamping::v2::comm_world>` and
{cpp:any}`comm_self() <kamping::v2::comm_self>` return views of them. It does not matter who initialized MPI, so a
library can call them as well; in debug builds, they check that MPI is initialized and not yet finalized.

```cpp
v2::environment env(argc, argv);
v2::comm_view   world = v2::comm_world();
```

In the **sessions model** (MPI 4), each part of a program opens its own {cpp:class}`kamping::v2::session` and
creates its communicators from *process sets*, such as `mpi://WORLD` and `mpi://SELF`
({cpp:any}`psets::world <kamping::v2::psets::world>`, {cpp:any}`psets::self <kamping::v2::psets::self>`). There is
no global initialization: a session sets up MPI only for the code that opened it, and the communicators are owned by
that code.

```cpp
v2::session s;
v2::comm    world = s.comm_from_pset(v2::psets::world, "my-lib");
```

`comm_from_pset` is collective over the process set. The tag (`"my-lib"`) must be the same on all processes; it
tells apart communicators that are created concurrently on several threads of a process, which must use different
tags. If you write a library, pass a tag unique to it, since you cannot know what the rest of the program does on
other threads.

We recommend sessions where MPI 4 is available: a library can set up MPI on its own, without global state and
without depending on how, or whether, the application initialized MPI. Sessions need MPI 4: KaMPIng-v2 provides
them if `mpi.h` reports MPI 4 or newer (`MPI_VERSION >= 4`), as MPICH 4 does. Open MPI 4.1, the default on Ubuntu
22.04, does not support sessions, and Open MPI 5 reports MPI 3.1. The world model therefore remains fully supported,
and both models can be used in the same program.

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

Ranks and tags accept any integer type, scoped enumerations, and types with an `mpi_rank()` / `mpi_tag()` member,
such as `periodic_rank` or the special values `any_source`, `any_tag` and `proc_null`. Scoped enumerations as tags
make message kinds type-safe:

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
