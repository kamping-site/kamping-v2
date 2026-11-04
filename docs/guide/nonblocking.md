# Non-blocking operations

A non-blocking operation starts a communication and returns before it has completed. MPI requires that its buffers
are left untouched until then — but the C interface cannot enforce this, and modifying, resizing or freeing a buffer
too early is a common source of bugs.

KaMPIng-v2 enforces it for the buffers it owns. Every operation takes its buffers by forwarding reference:

- a buffer passed as an **rvalue** is moved into the operation, which owns it until the operation has completed;
- a buffer passed as an **lvalue** is borrowed; you keep ownership and the responsibility not to touch it early.

A non-blocking operation returns an `iresult`, which holds the request and, for an rvalue, the buffer. The buffer
is only accessible through `wait()`, which completes the request and hands the buffer back, or through `test()`,
which hands it back only if the request has completed:

```cpp
std::vector<int> rbuf(n);
auto pending = v2::irecv(std::move(rbuf), source, comm);
// ... rbuf has been moved from and cannot be used by accident ...
rbuf = pending.wait();
```

```cpp
auto pending = v2::isend(std::move(data), dest, comm);
while (true) {
    if (auto done = pending.test()) {   // std::optional, holds the buffer once complete
        data = std::move(*done);
        break;
    }
    do_other_work();
}
```

Moving an `iresult` keeps the buffer where it is, so the pointer MPI is writing to stays valid.

## Completing requests

An `iresult` must be completed with `wait()` or a successful `test()`. The library never completes a request
implicitly: destroying an `iresult` whose operation is still active — or assigning over it — prints a diagnostic
and calls `std::terminate()`. This happens before the owned buffer is released, so a buffer that MPI may still be
writing to is never freed. Install a handler with `std::set_terminate` to customize what happens, e.g. to call
`MPI_Abort`.

The same rule applies to `request` and `request_vector` (see {doc}`../reference/results`), which you can use to
manage requests yourself: pass a request as the first argument of a non-blocking operation, and complete many at
once with `waitall`, `waitany` or `waitsome`.

```cpp
v2::request_vector requests;
for (int i = 0; i < n; ++i) {
    v2::isend(requests.emplace_back(), messages[i], dest[i], comm);
}
v2::waitall(requests);
```

## Blocking operations

Blocking operations follow the same ownership model. They return the buffers they were given — a single buffer
directly, a send and a receive buffer as a `result` — so a buffer that was moved in comes back out:

```cpp
auto data = v2::recv(std::vector<int>{} | views::resize, source, comm);
auto [_, all] = v2::allgatherv(mine, v2::auto_recv_v<int>(), comm);
```
