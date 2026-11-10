# Design: a v2 measurements suite (timers, counters, and beyond)

Motivation: the v1 `kamping::measurements` framework (`Timer`, `Counter`,
`AggregatedTree`, printers) predates the v2 architecture. It is templated on the
concrete v1 `Communicator`, built on v1 named-parameter collectives, dispatches
global aggregation through a closed `GlobalAggregationMode` enum, and — most
importantly — **embeds collective calls inside its print-time tree traversal**,
which makes it deadlock/corrupt whenever the per-rank measurement trees diverge
(a real problem in non-synchronized, data-dependent code).

This document specifies a ground-up rewrite that fits the v2 layering and,
crucially, rests on a single algebraic idea that makes local aggregation,
cross-rank aggregation, thread aggregation, and divergence handling all the same
mechanism — one that plugs directly into the existing `make_op` machinery.

---

## North Star

> One primitive (a **source**), one algebra (a **mergeable aggregator**), three
> decoupled phases (**record → reduce → present**). Zero overhead when off,
> near-zero when on, divergence-native by construction, and every reduction is
> literally an `MPI_Op`.

Non-goals (an optimal library is defined by its refusals too):

- **Not a sampling profiler / not Score-P.** Deterministic, in-code
  instrumentation only. Export to Perfetto; don't build a GUI or auto-trace
  every function.
- **No required dependencies.** Core is header-only over MPI. PAPI, RAPL energy,
  Perfetto export are ecosystem adapters.
- **No dynamic string work on the hot path** unless the user explicitly opts in.

---

## The spine: aggregators are commutative monoids → they *are* `MPI_Op`s

The whole design hangs off one concept.

```cpp
template <class A>
concept aggregator = requires(A a, A const b, typename A::input v) {
    A{};              // identity — the "empty"/absent element
    a.add(v);         // fold one local sample
    a.merge(b);       // combine two partial results (associative + commutative)
    a.value();        // final read-out (may differ from the internal state)
};
```

That is the entire theory. Everything else falls out:

| v1 concept                          | v2 realization                                  |
|-------------------------------------|-------------------------------------------------|
| `LocalAggregationMode` (accumulate/append) | repeated `add`                          |
| `GlobalAggregationMode` (min/max/sum/gather) enum + `switch` | a choice of `aggregator` type |
| cross-rank aggregation              | `merge`, fed to `make_op(&A::merge)`            |
| hybrid MPI+threads                  | merge per-thread aggregators, then per-rank     |
| "region missing on this rank"       | the identity element `A{}` (see below)          |

Because `merge` is associative and commutative, it is exactly the reduction
operator KaMPIng already knows how to lift into MPI via
`kamping::v2::make_op<A>(merge, commutative)`. So **a distributed aggregate is
one `MPI_Reduce`** — O(log p), fixed memory, no gather-to-root.

### Most useful aggregators are fixed-size PODs

```cpp
struct stats {                     // Welford: mergeable mean/stddev/min/max/argmax
    using input = double;
    std::uint64_t n = 0;
    double        mean = 0.0, m2 = 0.0;
    double        min  = +INFINITY, max = -INFINITY;
    int           argmax_rank = -1;   // which rank held the max — straggler id
    void add(double x);
    void merge(stats const&);         // parallel (chunked) Welford
    // value() → { n, mean, stddev = sqrt(m2/n), min, max, argmax_rank }
};
```

A `stats` is trivially copyable, so a `std::vector<stats>` gets an MPI datatype
for free via `kamping::types` (the `trivially_copyable` byte-blob adapter —
`sizeof(stats)` bytes as `MPI_BYTE`), and `make_op<stats>` interprets the blob.
No hand-written `MPI_Type_create_struct`.

Bounded-size aggregators (a fixed-capacity `t-digest`/HdrHistogram for exact
quantiles, or a reservoir sample) are *also* trivially mergeable and fixed-size,
so quantiles ride the same `MPI_Reduce`. Only the honest **`gather`** aggregator
(keep every value) is variable-size and needs a `gatherv` — it is the one
exception, not the rule. v1 gathered-to-root for *every* mode.

### Composition

Aggregators compose into a product monoid (still a monoid):

```cpp
auto a = agg::stats{} & agg::quantiles{0.5, 0.99} & agg::gather{};
// add()  → add to each; merge() → merge each; value() → tuple of values
```

---

## Divergence dissolves into the identity element

The v1 failure mode: `AggregatedTree::aggregate()` recurses the tree and calls
`comm.gatherv(...)` **per node**, and child order is first-`start()` order.
Non-synchronized code visits regions in data-dependent order → rank 0's Nth
`gatherv` pairs with rank 1's Nth on a *different* node → wrong data or deadlock.
The `is_same_on_all_ranks` asserts only *detect* this and abort.

The rewrite decouples communication from traversal entirely:

1. **Union the region paths.** Each rank hashes its set of full paths
   (`"algorithm/round/solve"`); one allgather-of-hashes + reconcile yields a
   canonical, globally-agreed ordering. (`present_count` below rides along.)
2. **Lay out `std::vector<A>` in canonical order**, with the **identity `A{}`
   wherever this rank has no data** for that path.
3. **One `MPI_Reduce` with `make_op<A>(merge)`.** `merge(x, A{})` is a no-op, so
   a rank that never entered a branch simply contributes identity. Divergence
   needs *zero* special-casing.

There is no lockstep traversal, no per-node collective, no deadlock. A
`present_count` aggregator (`add → +1`, `merge → +`) travels in the same reduce,
so the report states scope honestly:

```
solve:   mean=1.20s  σ=0.30  max=2.10s @rank37   6/8 ranks
```

Divergence becomes **reported data and straggler identification**, not a fatal
assert.

### Scaling note

Gather-to-root (serialize whole trees) also solves divergence but is O(P·tree)
at root. The reduce-over-fixed-size-aggregators approach is O(log P) and
bounded, and is strictly preferred *because* the identity element absorbs
missing regions. Reserve the plain `gatherv` path for the `gather` aggregator
only.

---

## Three phases, cleanly separated (the v2 layering)

v1's cardinal sin is entangling record, communicate, and print. Split them onto
the v2 layers:

| Phase       | Layer                              | Property                                              |
|-------------|------------------------------------|-------------------------------------------------------|
| **Record**  | `kamping::v2::measure`             | lock-free per thread, no comm, no hot-path allocation |
| **Reduce**  | `mpi::experimental`                | path-union + one `make_op` reduce; `handle(comm)`-based |
| **Present** | `kamping::v2::measure::exporters`  | consumes a *range* of aggregated nodes; pure, no MPI  |

Because **record never communicates**, a second sink becomes possible: instead
of folding into aggregators, log raw `(region, t_start, t_stop, thread)` events
and export a **Perfetto/Chrome trace** — timeline + flamegraph from the *same*
instrumentation. Aggregating mode and tracing mode are two sinks behind one
recording API.

Presentation consumes ranges, so exporters are trivial and additive:

```cpp
auto report = t.aggregate(comm);          // or aggregate_async(comm) → overlaps teardown
report | measure::to_table(std::cout);    // box-drawing / markdown
report | measure::to_json(out, manifest);
report | measure::to_perfetto("run.pb");
report | measure::to_flamegraph("prof.svg");
```

---

## Sources: one tree, many meters

The **source** is the second customization point (mirroring
`buffer_traits`/`handle_traits`).

```cpp
template <class S>
concept source = requires(S s, typename S::sample_type a, typename S::sample_type b) {
    typename S::sample_type;            // e.g. a time_point or a counter reading
    typename S::value_type;             // what a measurement yields (double seconds, int64 …)
    { s.sample() }        -> std::same_as<typename S::sample_type>;
    { s.delta(a, b) }     -> std::same_as<typename S::value_type>;   // b - a
};
```

`Timer` and `Counter` stop being separate classes:

```cpp
using timer   = tree<source::wall_clock>;          // MPI_Wtime
using counter = tree<source::value_counter<std::int64_t>>;
```

New meters plug in non-intrusively and compose on one region: PAPI hardware
counters, RAPL **energy**, allocated bytes, a user's domain metric.

```cpp
auto t = tree{source::wall_clock{}, source::papi{PAPI_L2_DCM}, source::energy_rapl{}};
```

`source::manual_clock` (a settable `now()`) makes the **entire framework
deterministically unit-testable** — something v1 cannot do because `MPI_Wtime`
is hardwired. This alone justifies the abstraction.

---

## Zero-overhead regions (the hot path)

v1 does a `std::string` hash + `unordered_map` lookup on *every* `start`/`stop`.
Kill it with compile-time keys + static registration:

```cpp
{
    KAMPING_MEASURE("assembly");   // static region token, registered once, ever
    ...                            // RAII scope: stops on destruction, exception-safe
}
```

`KAMPING_MEASURE` expands to a `static` region descriptor resolved on first pass
(a `fixed_string` NTTP → stable integer index into a flat array). Steady state =
one clock read + one array-indexed `add`: no hashing, no map, no allocation.
Disabled at compile time, the macro is `((void)0)` and evaporates. A runtime
`t.region(dynamic_key)` remains for genuinely dynamic keys at the old cost.

`KAMPING_MEASURE_AS("solve", agg::stats{} & agg::quantiles{...})` picks custom
aggregators for a region; the default is `agg::stats{}`.

---

## The features that make it *the* HPC library

- **Auto-instrumentation of KaMPIng itself.** A thin hook around
  `kamping::v2::` collectives records `{bytes, time, count}` per call site for
  free. One flag → every `alltoallv`/`allgatherv` measured, bytes *and* time,
  with no user annotation. No external profiler can do this because they are not
  *inside* the MPI wrapper. This is the unfair advantage.
- **Straggler/imbalance first-class.** `argmax_rank` and load-imbalance ratio
  (max/mean) answer HPC's number-one question — *who is slow, and where* — in
  the default table.
- **Multi-run manifests.** Each report carries
  `{comm_size, hostname, git_hash, timestamp, config}`; reports are themselves
  mergeable, so scaling series and run-to-run diffs are programmatic.
- **Mockable sources.** `manual_clock` → deterministic tests of the whole
  pipeline.

---

## API, end to end

```cpp
auto& t = measure::timer(comm);                        // any comm: handle() dispatch

t.synchronized("algorithm", [&]{                       // barrier-fenced scope
    for (int r = 0; r < rounds; ++r) {
        KAMPING_MEASURE("round");                      // zero-overhead region
        exchange(comm);                                //   auto-instrumented: bytes+time free
        {
            KAMPING_MEASURE_AS("solve",                // custom aggregators
                agg::stats{} & agg::quantiles{0.5, 0.99});
            solve();
        }
    }
});

auto report = t.aggregate(comm);                       // path-union + one make_op reduce
if (comm.is_root()) report | measure::to_table(std::cout)
                          | measure::to_perfetto("run.pb");
```

---

## Layout

```
mpi-core/include/mpi/measure/
    aggregator.hpp        // aggregator concept, make_op glue for merge
    reduce_tree.hpp       // path-union + MPI_Reduce<vector<A>>; handle()-based
kamping-v2/include/kamping/v2/measure/
    source.hpp            // source concept, wall_clock, manual_clock, value_counter
    aggregators.hpp       // stats (Welford), quantiles, gather, present_count, product &
    region.hpp            // region registry, RAII scope, KAMPING_MEASURE macros
    tree.hpp              // tree<Source...>, timer/counter presets, synchronized()
    exporters/            // to_table, to_json, to_perfetto, to_flamegraph
ecosystem/measure/
    papi/  energy_rapl/   // optional sources
```

---

## MVP staging

The design rests on one idea, so build and prove that first:

1. **`aggregator` concept + `stats` (Welford) + `make_op<stats>(merge)` reduce.**
   Validates the spine against the existing ops layer immediately.
2. **`source` concept + `wall_clock` + `manual_clock`**, unifying Timer/Counter
   and giving deterministic tests from day one.
3. **Path-union + identity-element reduce** → divergence-safe aggregation with
   `present_count`.
4. **`KAMPING_MEASURE` zero-overhead regions.**
5. **Sinks** (`to_table`, `to_perfetto`) + the **auto-instrumentation** hook —
   the headline feature.

Everything after (PAPI/energy sources, quantile sketches, multi-run diffing) is
additive and non-intrusive because the concepts are the contract.

See the companion sketches in [`measurements/`](measurements/) —
`aggregator.hpp` and `source.hpp` — for the concept headers and the `stats`
monoid as they would read against `make_op`.
