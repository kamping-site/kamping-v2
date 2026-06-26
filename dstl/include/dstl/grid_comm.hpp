// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <numeric>
#include <ranges>
#include <vector>

#include <mpi.h>

#include "dstl/factoring.hpp"
#include "dstl/tags.hpp"
#include "kamping/kassert/kassert.hpp"
#include "mpi/comm.hpp"
#include "mpi/environment.hpp"
#include "mpi/thread_level.hpp"

/// @file
/// dstl::grid_comm — a k-dimensional grid communicator built from k iterated splits of a
/// global communicator. Each rank is a mixed-radix coordinate vector (c_0, ..., c_{k-1}) with
/// c_0 the *most* significant digit (row-major): dimension 0 has the largest stride
/// (s_1·…·s_{k-1}), so subcommunicator 0 groups the most "remote" ranks — {0, s_1·…·s_{k-1}, …} —
/// while the *last* dimension has stride 1 and groups consecutive ranks. This row-major layout makes
/// the global-rank order coincide with the radix count-tree's leaf order, so dstl::grid_alltoallv routes
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
// dstl::grid_alltoallv routes over these subcommunicators in natural radix order, REMOTE-first
// (dimension 0 → k-1): phase 0 exchanges within {0,3}/{1,4}/{2,5} (fixing c0), phase 1 within
// {0,1,2}/{3,4,5} (fixing c1). Because the layout is row-major, the global-rank order already equals
// the count-tree's leaf order, so no top-level reorder is needed.
//
/// k-dimensional grid communicator. `k` (the number of dimensions) is a runtime value derived
/// from the factoring strategy; the execution policy `Exec` is a compile-time tag (D2/D3).
template <is_execution_policy Exec = execution_policy::par>
class grid_comm : public mpi::experimental::comm_accessors<grid_comm<Exec>> {
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

    /// Build a grid over `global` with explicit dimensions. `dims` is any input range of values
    /// convertible to `std::size_t`; the product of `dims` must equal the size of `global` (a complete grid).
    template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_value_t<R>, std::size_t>
    grid_comm(mpi::experimental::comm_view global, R&& dims) : _global(global) {
        if constexpr (std::ranges::sized_range<R>) {
            _dims.reserve(std::ranges::size(dims));
        }
        for (auto const& d: dims) {
            _dims.push_back(static_cast<std::size_t>(d));
        }
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

    /// @return The underlying global `MPI_Comm` (drives `comm_accessors` and `handle()` dispatch, so
    /// `.rank()`, `.size()`, `.group()`, `.dup()`, `.split()` operate on the global communicator).
    [[nodiscard]] MPI_Comm mpi_handle() const noexcept {
        return _global.mpi_handle();
    }

    /// Decompose a global rank into its mixed-radix coordinates (c_0 most significant), writing the
    /// `num_dims()` coordinates into the output range `out`. For a sized range, asserts it holds exactly
    /// `num_dims()` elements; for an unbounded sink (e.g. `std::views::counted` of a back-inserter) the
    /// caller is responsible for capacity.
    /// @return The iterator one past the last written coordinate.
    template <std::ranges::output_range<std::size_t> R>
    std::ranges::borrowed_iterator_t<R> coords(int rank, R&& out) const {
        if constexpr (std::ranges::sized_range<R>) {
            KAMPING_V2_ASSERT(std::ranges::size(out) == _dims.size(), "coords output range must have size num_dims()");
        }
        auto it = std::ranges::begin(out);
        auto r  = static_cast<std::size_t>(rank);
        for (std::size_t i = 0; i < _dims.size(); ++i) {
            *it++ = (r / _strides[i]) % _dims[i];
        }
        return it;
    }

    /// @return The mixed-radix coordinates of a global rank as a vector.
    [[nodiscard]] std::vector<std::size_t> coords(int rank) const {
        std::vector<std::size_t> out(_dims.size());
        coords(rank, out);
        return out;
    }

    /// Compose mixed-radix coordinates (c_0 most significant) back into a global rank. `coords` is any
    /// input range of exactly `num_dims()` values yielding the coordinates in dimension order.
    template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_value_t<R>, std::size_t>
    [[nodiscard]] int rank_of(R&& coords) const {
        if constexpr (std::ranges::sized_range<R>) {
            KAMPING_V2_ASSERT(std::ranges::size(coords) == _dims.size(), "coords range must have size num_dims()");
        }
        std::size_t r = 0;
        std::size_t i = 0;
        for (std::size_t const c: coords) {
            KAMPING_V2_ASSERT(i < _dims.size(), "coords range must have size num_dims()");
            r += c * _strides[i++];
        }
        KAMPING_V2_ASSERT(i == _dims.size(), "coords range must have size num_dims()");
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
        KAMPING_V2_ASSERT(!_dims.empty(), "grid must have at least one dimension");
        std::size_t product = std::accumulate(_dims.begin(), _dims.end(), std::size_t{1}, std::multiplies<>{});
        KAMPING_V2_ASSERT(
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
    }

    static void check_thread_level() {
        if constexpr (std::is_same_v<Exec, execution_policy::seq>) {
            return;
        } else {
            using mpi::experimental::ThreadLevel;
            ThreadLevel const required =
                std::is_same_v<Exec, execution_policy::par_comm> ? ThreadLevel::multiple : ThreadLevel::funneled;
            KAMPING_V2_ASSERT(
                mpi::experimental::environment::thread_level() >= required,
                "grid_comm: the MPI runtime does not provide the thread support level required by this execution policy"
            );
        }
    }

    mpi::experimental::comm_view         _global;
    std::vector<std::size_t>             _dims;
    std::vector<std::size_t>             _strides;
    std::vector<mpi::experimental::comm> _subcomms;
};

} // namespace dstl
