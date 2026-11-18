# Debugging

`<dstl/debugging.hpp>` stops selected processes until a debugger is attached to them, so that you can debug a run
from its start without launching it under a debugger.

## Waiting for a debugger

Call `wait_for_debugger_if_requested` right after initializing MPI and select the ranks to stop in the environment
variable `DSTL_WAIT_FOR_DEBUGGER`, as a comma-separated list of ranks and ranges, or `all`:

```cpp
MPI_Init(&argc, &argv);
dstl::wait_for_debugger_if_requested();
```

```text
$ mpirun -n 8 -x DSTL_WAIT_FOR_DEBUGGER=0,4-7 ./app
dstl: rank 0 (PID 4711 on node01) waiting for a debugger
...
$ gdb -p 4711     # or: lldb -p 4711
```

Each selected rank prints its PID and host and waits until a debugger is attached to it; the other ranks wait in a
barrier. Attaching stops the process: set your breakpoints and continue. Once every selected rank has its debugger,
all ranks continue together. If the variable is not set, the call does nothing, so it can stay in the code.

To choose the ranks in the code instead, call `wait_for_debugger` with them.

The wait needs to detect the debugger, which works on Linux and macOS. On other platforms, release a waiting process
from the debugger by setting the variable `keep_waiting` in its frame to 0 (gdb: `set var keep_waiting = 0`).

### When to use something else

- With a terminal on the node, you can start ranks under a debugger directly, without changing the code:
  `mpirun -n 1 gdb --args ./app : -n 7 ./app` debugs rank 0.
- Parallel debuggers such as Linaro DDT or TotalView start or attach to all ranks themselves.
- To inspect a run that hangs, attach to the hanging process (`gdb -p <pid>`); it does not need to wait.

```{doxygenoverloads} dstl::wait_for_debugger
```

```{doxygenoverloads} dstl::wait_for_debugger_if_requested
```

## Debugging the calling process

The counterparts of C++26 `<debugging>`. They call the standard functions where the standard library provides them.

```{doxygenoverloads} dstl::is_debugger_present
```

```{doxygenoverloads} dstl::breakpoint
```

```{doxygenoverloads} dstl::breakpoint_if_debugging
```
