# dstl

dstl is a library of distributed algorithms and communication patterns built on KaMPIng-v2. Its functions take the
same buffers and views as KaMPIng-v2, in namespace `dstl`. Include `<dstl/dstl.hpp>` and link `dstl::dstl` (see
{doc}`../guide/getting-started`).

Unlike the `kamping::v2` operations, dstl functions take the communicator as a default argument,
`MPI_COMM_WORLD`, after their required arguments.

## Sparse all-to-all

In many applications, each process exchanges data with only a few others, which are not known in advance. An
`alltoallv` would exchange counts with all processes; {doc}`sparse_alltoall <sparse-alltoall>` sends only the
messages that exist and lets each process discover what it receives, using one of three protocols from Hoefler,
Siebert and Lumsdaine (NBX, PCX, RSX):

```cpp
std::vector<std::pair<int, std::vector<int>>> sends = ...;   // (destination, message)
std::vector<int> received;
dstl::sparse_alltoall(sends, received | views::resize, tag, comm);
```

A callback overload gives you each message as it is matched, together with its source:

```cpp
dstl::sparse_alltoall(sends, [&](v2::probe_result probe) {
    int source = probe.source();
    auto msg   = v2::mrecv(std::vector<int>{} | views::resize, std::move(probe));
    // ...
}, tag, comm);
```

## All-to-all-v on many processes

{doc}`alltoallv` covers two drop-in replacements for `kamping::v2::alltoallv`:

- `grid_alltoallv` routes the data through a grid of processes, which reduces the number of messages per process
  from p−1 to about 2√p. This pays off for many small messages on many processes.
- `alltoallv` on a `thread_multiple_comm` uses several threads for the communication itself.

## Distributions and redistribution

A `distribution` describes how a global array is split into contiguous parts over the processes, and
`redistribute` moves data from one such split to another. See {doc}`redistribute`.

## Request-reply

`request_reply` sends requests to the processes that own the data, answers them there and returns the replies, as
in a distributed lookup. See {doc}`request-reply`.

## Algorithms

Distributed counterparts of standard algorithms on a range distributed over the processes: `reduce`, `min`,
`max`, `all_of`, `is_sorted`, prefix sums and shifts. See {doc}`algorithms`.
