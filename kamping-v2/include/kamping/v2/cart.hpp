// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <vector>

#include <mpi.h>

#include "kamping/v2/comm.hpp"
#include "mpi/cart.hpp"
#include "mpi/handle.hpp"
#include "mpi/rank.hpp"

/// @file
/// Cartesian process topologies in `kamping::v2`: the core functions from `mpi/cart.hpp`, plus a form of
/// `cart_create` that takes one descriptor per dimension.

namespace kamping::v2 {
using mpi::experimental::auto_dim;
using mpi::experimental::cart_coords;
using mpi::experimental::cart_create;
using mpi::experimental::cart_get;
using mpi::experimental::cart_position;
using mpi::experimental::cart_rank;
using mpi::experimental::cart_shift;
using mpi::experimental::cart_shift_result;
using mpi::experimental::cart_sub;
using mpi::experimental::cartdim;
using mpi::experimental::dims_create;
using mpi::experimental::flag_array;

/// @brief A dimension of `cart_create`: `procs` processes, periodic (wrapping around) or not.
///
/// `procs` may be `auto_dim`: `cart_create` then chooses it with `dims_create`, so that the grid covers the whole
/// communicator.
struct cart_dim {
    int  procs;            ///< Processes in this dimension, or `auto_dim`.
    bool periodic = false; ///< Whether the dimension wraps around.
};

/// @brief Collective: creates a communicator with a Cartesian topology, one `cart_dim` per dimension
/// (`MPI_Cart_create`). With `reorder` (`false` if left out), MPI may give a process a different rank in the grid
/// than in `c`.
///
/// @code
/// auto grid = v2::cart_create(world, {{v2::auto_dim, true}, {v2::auto_dim, true}});  // balanced, periodic
/// @endcode
///
/// Dimensions with `procs == auto_dim` are chosen by `dims_create`. A grid with fewer processes than `c` leaves the
/// other processes with a null `comm`.
/// @return The new communicator, owned.
/// @throws mpi_error if `MPI_Dims_create` or `MPI_Cart_create` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] comm cart_create(Comm const& c, std::span<cart_dim const> dims, bool reorder) {
    std::vector<int> procs;
    std::vector<int> periods;
    procs.reserve(dims.size());
    periods.reserve(dims.size());
    for (auto const& d: dims) {
        procs.push_back(d.procs);
        periods.push_back(d.periodic ? 1 : 0);
    }
    if (std::ranges::find(procs, auto_dim) != procs.end()) {
        mpi::experimental::dims_create(
            mpi::experimental::detail::comm_size(mpi::experimental::handle(c)),
            std::span<int>(procs)
        );
    }
    return mpi::experimental::cart_create(c, std::span<int const>(procs), periods, reorder);
}

/// @brief Collective: creates a communicator with a Cartesian topology, one `cart_dim` per dimension
/// (`MPI_Cart_create`). With `reorder` (`false` if left out), MPI may give a process a different rank in the grid
/// than in `c`.
/// @return The new communicator, owned.
/// @throws mpi_error if `MPI_Dims_create` or `MPI_Cart_create` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] comm cart_create(Comm const& c, std::initializer_list<cart_dim> dims, bool reorder) {
    return kamping::v2::cart_create(c, std::span<cart_dim const>(dims.begin(), dims.size()), reorder);
}

/// @brief Collective: creates a communicator with a Cartesian topology, one `cart_dim` per dimension
/// (`MPI_Cart_create`). With `reorder` (`false` if left out), MPI may give a process a different rank in the grid
/// than in `c`.
/// @return The new communicator, owned.
/// @throws mpi_error if `MPI_Dims_create` or `MPI_Cart_create` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] comm cart_create(Comm const& c, std::span<cart_dim const> dims) {
    return kamping::v2::cart_create(c, dims, false);
}

/// @brief Collective: creates a communicator with a Cartesian topology, one `cart_dim` per dimension
/// (`MPI_Cart_create`). With `reorder` (`false` if left out), MPI may give a process a different rank in the grid
/// than in `c`.
/// @return The new communicator, owned.
/// @throws mpi_error if `MPI_Dims_create` or `MPI_Cart_create` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] comm cart_create(Comm const& c, std::initializer_list<cart_dim> dims) {
    return kamping::v2::cart_create(c, dims, false);
}

/// @return The grid of the Cartesian communicator `c`, one `cart_dim` per dimension (`MPI_Cart_get`), e.g. for the
/// `procs` of a `darray_dim`.
/// @throws mpi_error if `MPI_Cartdim_get` or `MPI_Cart_get` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] std::vector<cart_dim> cart_dims(Comm const& c) {
    auto const       n = static_cast<std::size_t>(mpi::experimental::cartdim(c));
    std::vector<int> procs(n), periods(n), coords(n);
    mpi::experimental::cart_get(c, procs, periods, coords);
    std::vector<cart_dim> dims;
    dims.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        dims.push_back(cart_dim{procs[i], periods[i] != 0});
    }
    return dims;
}

} // namespace kamping::v2
