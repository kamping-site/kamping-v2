// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <functional>
#include <numeric>
#include <span>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <mpi.h>

#include "dstl/factoring.hpp"
#include "dstl/tags.hpp"
#include "kamping/kassert/kassert.hpp"
#include "mpi/comm.hpp"

/// @file
/// dstl::grid_comm — a k-dimensional grid communicator built from k iterated splits of a
/// global communicator. Each rank is a mixed-radix coordinate vector (c_0, ..., c_{k-1}) with
/// c_0 the *most* significant digit (row-major): dimension 0 has the largest stride
/// (s_1·…·s_{k-1}), so subcommunicator 0 groups the most "remote" ranks — {0, s_1·…·s_{k-1}, …} —
/// while the *last* dimension has stride 1 and groups consecutive ranks. This row-major layout makes
/// the global-rank order coincide with the radix count-tree's leaf order, so dstl::alltoallv routes
/// remote-first with no top-level reordering. Subcommunicator i groups all ranks that share every
/// coordinate except coordinate i and therefore has size s_i.
///
/// Mental model: picture the compute nodes as the *columns* of the grid. Ranks are consecutive
/// within a node (a column), i.e. along the last, stride-1 dimension; the first, largest-stride
/// dimension steps across nodes (columns). See DSTL-Alltoallv-Design.md (§5–§7).

namespace dstl {

// How ranks map onto the grid and its subcommunicators (k = 2 example)//
// p = 6, dims = [2, 3]  (s0 = 2 = #nodes, s1 = 3 = ranks/node); row-major, c0 most significant →
// stride = [3, 1]. Picture the compute NODES as the COLUMNS; ranks are consecutive *down* a column.
//
//   Global rank r  →  coords (c0, c1):   c0 = r / 3 (node),  c1 = r % 3 (rank within node)
//
//            c0=0      c0=1        ← columns = compute nodes (dimension 0, remote, stride 3)
//            node 0    node 1
//    c1=0 │    0         3
//    c1=1 │    1         4
//    c1=2 │    2         5
//             └ consecutive ranks run down each column (node)
//
//   subcomm 0 (dimension 0, size 2, stride 3): {0,3} {1,4} {2,5}   ← inter-node (one rank per node)
//   subcomm 1 (dimension 1, size 3, stride 1): {0,1,2} {3,4,5}     ← intra-node (consecutive ranks)
//
// dstl::alltoallv routes over these subcommunicators in natural radix order, REMOTE-first
// (dimension 0 → k-1): phase 0 exchanges within {0,3}/{1,4}/{2,5} (fixing c0), phase 1 within
// {0,1,2}/{3,4,5} (fixing c1). Because the layout is row-major, the global-rank order already equals
// the count-tree's leaf order, so no top-level reorder is needed.
//
/// k-dimensional grid communicator. `k` (the number of dimensions) is a runtime value derived
/// from the factoring strategy; the execution policy `Exec` is a compile-time tag (D2/D3).
template <execution_policy Exec = funneled>
class grid_comm {
public:
    using execution_policy_type = Exec;

    /// Build a grid over `global` using the default factoring strategy (balanced 2-level grid).
    explicit grid_comm(mpi::experimental::comm_view global) : grid_comm(global, dims_create{}) {}

    /// Build a grid over `global` using a factoring strategy.
    template <factoring Strategy>
    grid_comm(mpi::experimental::comm_view global, Strategy strategy) : _global(global),
                                                                        _dims(strategy(global)) {
        build();
    }

    /// Build a grid over `global` with explicit dimensions. The product of `dims` must equal
    /// the size of `global` (a complete grid).
    grid_comm(mpi::experimental::comm_view global, std::span<std::size_t const> dims)
        : _global(global),
          _dims(dims.begin(), dims.end()) {
        build();
    }

    grid_comm(grid_comm const&)            = delete;
    grid_comm& operator=(grid_comm const&) = delete;
    grid_comm(grid_comm&&)                 = default;
    grid_comm& operator=(grid_comm&&)      = default;
    ~grid_comm()                           = default;

    /// @return The number of grid dimensions k.
    [[nodiscard]] std::size_t num_dims() const noexcept {
        return _dims.size();
    }

    /// @return The size s_i of dimension i.
    [[nodiscard]] std::size_t dim_size(std::size_t i) const {
        return _dims[i];
    }

    /// @return A non-owning view of the subcommunicator for dimension i (size s_i).
    [[nodiscard]] mpi::experimental::comm_view subcomm(std::size_t i) const {
        return _subcomms[i];
    }

    /// @return The number of per-thread duplicated communicator sets. For the `thread_multiple`
    /// policy this is the OpenMP thread count fixed at construction (`omp_get_max_threads()`);
    /// for every other policy it is 1.
    [[nodiscard]] std::size_t num_threads() const noexcept {
        if constexpr (std::is_same_v<Exec, thread_multiple>) {
            return _thread_subcomms.empty() ? 1 : _thread_subcomms.front().size();
        } else {
            return 1;
        }
    }

    /// @return A non-owning view of OpenMP thread `thread`'s private duplicate of the
    /// subcommunicator for dimension `dim`. Only available for the `thread_multiple` policy, which
    /// owns one duplicated set of subcommunicators per thread so that the per-phase MPI exchanges
    /// can run concurrently, each thread on its own communicator.
    [[nodiscard]] mpi::experimental::comm_view subcomm(std::size_t dim, std::size_t thread) const {
        static_assert(
            std::is_same_v<Exec, thread_multiple>,
            "per-thread subcommunicators only exist for the thread_multiple execution policy"
        );
        return _thread_subcomms[dim][thread];
    }

    /// @return A view of the underlying global communicator.
    [[nodiscard]] mpi::experimental::comm_view global() const noexcept {
        return _global;
    }

    /// @return The calling rank in the global communicator.
    [[nodiscard]] int rank() const {
        return _global.rank();
    }

    /// @return The global communicator size p.
    [[nodiscard]] int size() const {
        return _global.size();
    }

    /// Decompose a global rank into its mixed-radix coordinates (c_0 most significant).
    void coords(int rank, std::span<std::size_t> out) const {
        KAMPING_ASSERT(out.size() == _dims.size(), "coords output span must have size num_dims()");
        auto r = static_cast<std::size_t>(rank);
        for (std::size_t i = 0; i < _dims.size(); ++i) {
            out[i] = (r / _strides[i]) % _dims[i];
        }
    }

    /// @return The mixed-radix coordinates of a global rank as a vector.
    [[nodiscard]] std::vector<std::size_t> coords(int rank) const {
        std::vector<std::size_t> out(_dims.size());
        coords(rank, out);
        return out;
    }

    /// Compose mixed-radix coordinates back into a global rank.
    [[nodiscard]] int rank_of(std::span<std::size_t const> coords) const {
        KAMPING_ASSERT(coords.size() == _dims.size(), "coords span must have size num_dims()");
        std::size_t r = 0;
        for (std::size_t i = 0; i < _dims.size(); ++i) {
            r += coords[i] * _strides[i];
        }
        return static_cast<int>(r);
    }

    /// @return True if the grid is complete (product of dims == p). Always true in this first
    /// cut, which rejects ragged grids at construction time.
    [[nodiscard]] bool is_complete() const noexcept {
        return true;
    }

    /// @return The row-major stride of dimension i (product of all later dimensions; 1 for i==k-1).
    [[nodiscard]] std::size_t stride(std::size_t i) const {
        return _strides[i];
    }

private:
    /// Shared construction: validate the (complete) grid, precompute row-major strides, and split off
    /// one subcommunicator per dimension.
    void build() {
        check_thread_level();
        auto p = static_cast<std::size_t>(_global.size());
        KAMPING_ASSERT(!_dims.empty(), "grid must have at least one dimension");
        std::size_t product = std::accumulate(_dims.begin(), _dims.end(), std::size_t{1}, std::multiplies<>{});
        KAMPING_ASSERT(
            product == p,
            "grid_comm: product of dimensions must equal the global communicator size (only complete grids are "
            "supported in this first cut)"
        );

        // Row-major strides: stride[i] = product of all later dimensions (stride[k-1] == 1).
        // Dimension 0 thus has the largest stride → subcommunicator 0 groups the most remote ranks,
        // and the global-rank order coincides with the radix count-tree's leaf order.
        _strides.assign(_dims.size(), 1);
        for (std::size_t i = _dims.size(); i-- > 1;) {
            _strides[i - 1] = _strides[i] * _dims[i];
        }

        // One subcommunicator per dimension.
        // TODO: replace MPI_Comm_split with MPI_Comm_create_group. The membership of every
        // subcommunicator is known locally and globally (it is a closed-form function of the
        // mixed-radix coordinates), so we can build each subgroup with MPI_Group_incl and create
        // its communicator with MPI_Comm_create_group — avoiding the global sort/agreement that
        // MPI_Comm_split performs and that the grid is specifically trying to avoid.
        int rank = _global.rank();
        _subcomms.reserve(_dims.size());
        for (std::size_t i = 0; i < _dims.size(); ++i) {
            auto coord_i = (static_cast<std::size_t>(rank) / _strides[i]) % _dims[i];
            int  color   = rank - static_cast<int>(coord_i * _strides[i]); // shared across the fiber
            int  key     = static_cast<int>(coord_i);
            _subcomms.push_back(_global.split(color, key));
        }

        if constexpr (std::is_same_v<Exec, thread_multiple>) {
            build_thread_subcomms();
        }
    }

    /// Duplicate every per-dimension subcommunicator once per OpenMP thread, so each thread can drive
    /// its phase exchanges on a private communicator (the `MPI_THREAD_MULTIPLE` model). `MPI_Comm_dup`
    /// is collective over each subcommunicator; this assumes a uniform `omp_get_max_threads()` across
    /// the ranks of every subcommunicator (matching the multi-threaded all-to-all reference design).
    void build_thread_subcomms() {
#ifdef _OPENMP
        auto const nthreads = static_cast<std::size_t>(omp_get_max_threads());
        _thread_subcomms.resize(_dims.size());
        for (std::size_t i = 0; i < _dims.size(); ++i) {
            _thread_subcomms[i].reserve(nthreads);
            for (std::size_t t = 0; t < nthreads; ++t) {
                _thread_subcomms[i].push_back(_subcomms[i].dup());
            }
        }
#else
        KAMPING_ASSERT(false, "grid_comm: the thread_multiple execution policy requires an OpenMP build");
#endif
    }

    static void check_thread_level() {
        if constexpr (std::is_same_v<Exec, sequential>) {
            return;
        } else {
            int provided = MPI_THREAD_SINGLE;
            MPI_Query_thread(&provided);
            int required = std::is_same_v<Exec, thread_multiple> ? MPI_THREAD_MULTIPLE : MPI_THREAD_FUNNELED;
            KAMPING_ASSERT(
                provided >= required,
                "grid_comm: the MPI runtime does not provide the thread support level required by this execution policy"
            );
        }
    }

    mpi::experimental::comm_view         _global;
    std::vector<std::size_t>             _dims;
    std::vector<std::size_t>             _strides;
    std::vector<mpi::experimental::comm> _subcomms;

    /// Per-thread duplicates of each subcommunicator, shaped `[dim][thread]`. Populated only for the
    /// `thread_multiple` policy (empty otherwise).
    std::vector<std::vector<mpi::experimental::comm>> _thread_subcomms;
};

} // namespace dstl
