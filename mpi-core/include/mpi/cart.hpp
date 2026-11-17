// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>
#include <ranges>
#include <span>
#include <vector>

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/comm.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"
#include "mpi/int_array.hpp"
#include "mpi/rank.hpp"

/// @file
/// Cartesian process topologies (`MPI_Dims_create`, `MPI_Cart_*`).
///
/// A Cartesian topology is a property of a communicator, not a separate kind of object: `cart_create` returns an
/// ordinary owning `comm`, and the functions below take any communicator with a Cartesian topology (asserted in debug
/// builds). The communicator types also have members that delegate to them: `grid.cart_coords()`,
/// `grid.cart_position()`, `grid.cart_shift(0, 1)`, `grid.cart_sub(1)`, ...
///
/// Logical arrays (`periods`, `remain_dims`) are taken as a contiguous range of `int`, passed to MPI as it is, or as a
/// range of `bool` (see `flag_array`).

namespace mpi::experimental {

/// @brief A dimension that `dims_create` chooses (0): `dims_create<2>(p, {auto_dim, 4})` fixes the second dimension
/// to 4 and chooses the first.
inline constexpr int auto_dim = 0;

namespace detail {
/// Asserts that `c` has a Cartesian topology.
inline void assert_cart([[maybe_unused]] MPI_Comm c) {
    KAMPING_V2_ASSERT(
        mpi::experimental::topo_test(c) == topology::cart,
        "the communicator has no Cartesian topology; create one with cart_create"
    );
}
} // namespace detail

// ── cart_position ──────────────────────────────────────────────────────────────────────────────────────────────────

namespace detail {
struct cart_position_access;
} // namespace detail

/// @brief A position in the grid of a Cartesian communicator, with arithmetic along its dimensions.
///
/// `shift(dim, disp)` moves the position `disp` steps along dimension `dim`: in a periodic dimension, it wraps around;
/// in a non-periodic one, a position beyond the ends stands for `MPI_PROC_NULL` but keeps its coordinates, so shifting
/// back returns into the grid. A position can be passed wherever a rank is expected.
///
/// @code
/// auto const me   = grid.cart_position();
/// auto const diag = me.shift(0, 1).shift(1, 1);   // the diagonal neighbor, if there is one
/// sendrecv(sbuf, diag, rbuf, me.shift(0, -1).shift(1, -1), grid);
/// @endcode
///
/// Its ranks come from MPI (`MPI_Cart_rank`, `MPI_Cart_shift`), computed when the position is created, so using it as
/// a rank needs no MPI call. A position refers to its communicator without owning it, like a `comm_view`, so the
/// communicator must outlive it.
class cart_position {
public:
    /// @brief The position of the process `rank` in the Cartesian communicator `c` (`MPI_Cart_get`,
    /// `MPI_Cart_coords`).
    /// @throws mpi_error if an MPI call fails.
    template <convertible_to_mpi_handle<MPI_Comm> Comm>
    cart_position(Comm const& c, int rank) : _comm(mpi::experimental::handle(c)),
                                             _rank(rank) {
        detail::assert_cart(_comm);
        KAMPING_V2_ASSERT(rank >= 0, "cart_position: rank must be a rank of the communicator");
        int ndims = 0;
        int err   = MPI_Cartdim_get(_comm, &ndims);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        auto const n = static_cast<std::size_t>(ndims);
        _dims.resize(n);
        _periods.resize(n);
        _coords.resize(n);
        err = MPI_Cart_get(_comm, ndims, _dims.data(), _periods.data(), _coords.data());
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        err = MPI_Cart_coords(_comm, rank, ndims, _coords.data());
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    }

    /// @brief The position of the calling process in the Cartesian communicator `c`.
    /// @throws mpi_error if an MPI call fails.
    template <convertible_to_mpi_handle<MPI_Comm> Comm>
    explicit cart_position(Comm const& c) : cart_position(c, comm_view(mpi::experimental::handle(c)).rank()) {}

    /// @return The position `disp` steps along dimension `dim` (`MPI_Cart_rank`).
    /// @throws mpi_error if `MPI_Cart_rank` or `MPI_Cart_coords` fails.
    [[nodiscard]] cart_position shift(int dim, int disp) const {
        KAMPING_V2_ASSERT(
            dim >= 0 && static_cast<std::size_t>(dim) < _coords.size(),
            "cart_position::shift: dim is not a dimension"
        );
        cart_position shifted = *this;
        shifted._coords[static_cast<std::size_t>(dim)] += disp;
        shifted.resolve();
        return shifted;
    }

    /// @return The coordinates, one per dimension. Beyond the ends of a non-periodic dimension, they lie outside of the
    /// grid.
    [[nodiscard]] std::span<int const> coords() const noexcept {
        return _coords;
    }

    /// @return Whether the position is outside of the grid, i.e. stands for `MPI_PROC_NULL`.
    [[nodiscard]] bool is_proc_null() const noexcept {
        return _rank == MPI_PROC_NULL;
    }

    /// @return The rank at this position, or `MPI_PROC_NULL` outside of the grid.
    [[nodiscard]] int value() const noexcept {
        return _rank;
    }

    /// @return `value()`, for the rank protocol (`to_rank`).
    [[nodiscard]] int mpi_rank() const noexcept {
        return _rank;
    }

    /// @return Whether both are the same position in the same communicator.
    friend bool operator==(cart_position const& lhs, cart_position const& rhs) noexcept {
        return lhs._comm == rhs._comm && lhs._coords == rhs._coords;
    }

private:
    friend struct detail::cart_position_access;

    /// Whether the coordinates lie beyond the ends of a non-periodic dimension, where MPI defines no rank.
    [[nodiscard]] bool outside_grid() const noexcept {
        for (std::size_t i = 0; i < _coords.size(); ++i) {
            if (_periods[i] == 0 && (_coords[i] < 0 || _coords[i] >= _dims[i])) {
                return true;
            }
        }
        return false;
    }

    /// Computes the rank from the coordinates (`MPI_Cart_rank`, which wraps around in periodic dimensions) and
    /// normalizes the coordinates to MPI's (`MPI_Cart_coords`).
    void resolve() {
        if (outside_grid()) {
            _rank = MPI_PROC_NULL;
            return;
        }
        int err = MPI_Cart_rank(_comm, _coords.data(), &_rank);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        err = MPI_Cart_coords(_comm, _rank, static_cast<int>(_coords.size()), _coords.data());
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    }

    MPI_Comm         _comm;
    int              _rank;
    std::vector<int> _dims;
    std::vector<int> _periods;
    std::vector<int> _coords;
};

/// @brief The neighbors `cart_shift` computes, as positions: `source` to receive from, `dest` to send to. Usable with
/// structured bindings: `auto [source, dest] = grid.cart_shift(0, 1);`
struct cart_shift_result {
    cart_position source; ///< The position `disp` steps back along the dimension.
    cart_position dest;   ///< The position `disp` steps ahead along the dimension.
};

namespace detail {
struct cart_position_access {
    /// `from` shifted by `disp` along `dim`, with the rank `rank` that MPI computed for it (`MPI_Cart_shift`).
    static cart_position shifted(cart_position const& from, int dim, int disp, int rank) {
        cart_position result = from;
        result._coords[static_cast<std::size_t>(dim)] += disp;
        result._rank = rank;
        if (rank != MPI_PROC_NULL) {
            int err =
                MPI_Cart_coords(result._comm, rank, static_cast<int>(result._coords.size()), result._coords.data());
            if (err != MPI_SUCCESS) {
                throw mpi_error(err);
            }
        }
        return result;
    }
};
} // namespace detail

// ── MPI_Dims_create ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief Chooses a balanced grid of `nnodes` processes with `dims.size()` dimensions (`MPI_Dims_create`).
///
/// Entries of `dims` that are `auto_dim` (0) are chosen, the others are kept; the product of the kept entries must
/// divide `nnodes`. The chosen entries are in non-increasing order.
/// @throws mpi_error if `MPI_Dims_create` fails.
inline void dims_create(int nnodes, std::span<int> dims) {
    int err = MPI_Dims_create(nnodes, static_cast<int>(dims.size()), dims.data());
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @return A balanced grid of `nnodes` processes with `N` dimensions (`MPI_Dims_create`), e.g.
/// `dims_create<2>(12)` is `{4, 3}`.
/// @throws mpi_error if `MPI_Dims_create` fails.
template <std::size_t N>
[[nodiscard]] std::array<int, N> dims_create(int nnodes) {
    std::array<int, N> dims{};
    mpi::experimental::dims_create(nnodes, std::span<int>(dims));
    return dims;
}

/// @return `dims` with every `auto_dim` (0) entry chosen so that the grid has `nnodes` processes and is balanced
/// (`MPI_Dims_create`), e.g. `dims_create(12, std::array{auto_dim, 2})` is `{6, 2}`.
/// @throws mpi_error if `MPI_Dims_create` fails.
template <std::size_t N>
[[nodiscard]] std::array<int, N> dims_create(int nnodes, std::array<int, N> dims) {
    mpi::experimental::dims_create(nnodes, std::span<int>(dims));
    return dims;
}

// ── MPI_Cart_create ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief Collective: creates a communicator with a Cartesian topology of `dims.size()` dimensions, `dims[i]`
/// processes in dimension `i` and periodic (wrapping) where `periods[i]` is true (`MPI_Cart_create`).
///
/// The grid may have fewer processes than `c`; the processes left over get `MPI_COMM_NULL`. With `reorder`, MPI may
/// give a process a different rank in the grid than in `c`.
/// @param new_comm Output: receives the new communicator.
/// @throws mpi_error if `MPI_Cart_create` fails.
template <convertible_to_mpi_handle<MPI_Comm> Comm, flag_array Periods, convertible_to_mpi_handle_ptr<MPI_Comm> NewComm>
void cart_create(Comm const& c, std::span<int const> dims, Periods const& periods, bool reorder, NewComm& new_comm) {
    KAMPING_V2_ASSERT(
        dims.size() == static_cast<std::size_t>(std::ranges::size(periods)),
        "cart_create: dims and periods must have the same length"
    );
    int err = detail::with_int_array<bool>(periods, [&](int const* periods_ptr) {
        return MPI_Cart_create(
            mpi::experimental::handle(c),
            static_cast<int>(dims.size()),
            dims.data(),
            periods_ptr,
            reorder ? 1 : 0,
            mpi::experimental::handle_ptr(new_comm)
        );
    });
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief Collective: creates a communicator with a Cartesian topology of `dims.size()` dimensions, `dims[i]`
/// processes in dimension `i` and periodic (wrapping) where `periods[i]` is true (`MPI_Cart_create`).
///
/// The grid may have fewer processes than `c`; the processes left over get a null `comm`. With `reorder`, MPI may
/// give a process a different rank in the grid than in `c`.
/// @return The new communicator, owned.
/// @throws mpi_error if `MPI_Cart_create` fails.
template <convertible_to_mpi_handle<MPI_Comm> Comm, flag_array Periods>
[[nodiscard]] comm cart_create(Comm const& c, std::span<int const> dims, Periods const& periods, bool reorder) {
    MPI_Comm out = MPI_COMM_NULL;
    mpi::experimental::cart_create(c, dims, periods, reorder, out);
    return comm::from_native(out);
}

// ── MPI_Cartdim_get / MPI_Cart_get ─────────────────────────────────────────────────────────────────────────────────

/// @return The number of dimensions of the Cartesian communicator `c` (`MPI_Cartdim_get`).
/// @throws mpi_error if `MPI_Cartdim_get` fails.
template <convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] int cartdim(Comm const& c) {
    detail::assert_cart(mpi::experimental::handle(c));
    int ndims = 0;
    int err   = MPI_Cartdim_get(mpi::experimental::handle(c), &ndims);
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return ndims;
}

/// @brief The grid of the Cartesian communicator `c` (`MPI_Cart_get`): the number of processes and the periodicity
/// (nonzero if periodic) of each dimension, and the coordinates of the calling process.
///
/// All three arrays must have `cartdim(c)` elements.
/// @throws mpi_error if `MPI_Cart_get` fails.
template <convertible_to_mpi_handle<MPI_Comm> Comm>
void cart_get(Comm const& c, std::span<int> dims, std::span<int> periods, std::span<int> coords) {
    detail::assert_cart(mpi::experimental::handle(c));
    KAMPING_V2_ASSERT(
        dims.size() == periods.size() && dims.size() == coords.size(),
        "cart_get: dims, periods and coords must have the same length"
    );
    int err = MPI_Cart_get(
        mpi::experimental::handle(c),
        static_cast<int>(dims.size()),
        dims.data(),
        periods.data(),
        coords.data()
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

// ── MPI_Cart_coords / MPI_Cart_rank ────────────────────────────────────────────────────────────────────────────────

/// @brief The coordinates of the process `rank` in the Cartesian communicator `c` (`MPI_Cart_coords`).
/// @param coords Output: receives the coordinates; must have `cartdim(c)` elements.
/// @throws mpi_error if `MPI_Cart_coords` fails.
template <convertible_to_mpi_handle<MPI_Comm> Comm>
void cart_coords(Comm const& c, int rank, std::span<int> coords) {
    detail::assert_cart(mpi::experimental::handle(c));
    int err = MPI_Cart_coords(mpi::experimental::handle(c), rank, static_cast<int>(coords.size()), coords.data());
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @return The coordinates of the process `rank` in the Cartesian communicator `c` (`MPI_Cart_coords`).
/// @throws mpi_error if `MPI_Cart_coords` fails.
template <convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] std::vector<int> cart_coords(Comm const& c, int rank) {
    std::vector<int> coords(static_cast<std::size_t>(mpi::experimental::cartdim(c)));
    mpi::experimental::cart_coords(c, rank, std::span<int>(coords));
    return coords;
}

/// @return The coordinates of the calling process in the Cartesian communicator `c` (`MPI_Cart_coords`).
/// @throws mpi_error if `MPI_Cart_coords` fails.
template <convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] std::vector<int> cart_coords(Comm const& c) {
    return mpi::experimental::cart_coords(c, comm_view(mpi::experimental::handle(c)).rank());
}

/// @return The coordinates of the process `rank` in the `N`-dimensional Cartesian communicator `c`
/// (`MPI_Cart_coords`).
/// @throws mpi_error if `MPI_Cart_coords` fails.
template <std::size_t N, convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] std::array<int, N> cart_coords(Comm const& c, int rank) {
    KAMPING_V2_ASSERT(
        mpi::experimental::cartdim(c) == static_cast<int>(N),
        "cart_coords<N>: the communicator does not have N dimensions"
    );
    std::array<int, N> coords{};
    mpi::experimental::cart_coords(c, rank, std::span<int>(coords));
    return coords;
}

/// @return The coordinates of the calling process in the `N`-dimensional Cartesian communicator `c`, e.g. for
/// structured bindings: `auto [row, col] = cart_coords<2>(grid);` (`MPI_Cart_coords`).
/// @throws mpi_error if `MPI_Cart_coords` fails.
template <std::size_t N, convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] std::array<int, N> cart_coords(Comm const& c) {
    return mpi::experimental::cart_coords<N>(c, comm_view(mpi::experimental::handle(c)).rank());
}

/// @return The rank of the process at `coords` in the Cartesian communicator `c` (`MPI_Cart_rank`). In a periodic
/// dimension, coordinates outside of the grid wrap around.
/// @throws mpi_error if `MPI_Cart_rank` fails.
template <convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] int cart_rank(Comm const& c, std::span<int const> coords) {
    detail::assert_cart(mpi::experimental::handle(c));
    KAMPING_V2_ASSERT(
        static_cast<int>(coords.size()) == mpi::experimental::cartdim(c),
        "cart_rank: coords must have one entry per dimension"
    );
    int rank = MPI_PROC_NULL;
    int err  = MPI_Cart_rank(mpi::experimental::handle(c), coords.data(), &rank);
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return rank;
}

/// @return The rank of the process at `coords` in the Cartesian communicator `c` (`MPI_Cart_rank`), e.g.
/// `cart_rank(grid, {row, col})`.
/// @throws mpi_error if `MPI_Cart_rank` fails.
template <convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] int cart_rank(Comm const& c, std::initializer_list<int> coords) {
    return mpi::experimental::cart_rank(c, std::span<int const>(coords.begin(), coords.size()));
}

// ── MPI_Cart_shift ─────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief The neighbors `disp` steps back (`source`) and ahead (`dest`) along dimension `direction` of the Cartesian
/// communicator `c` (`MPI_Cart_shift`). In a non-periodic dimension, there is no neighbor beyond the ends
/// (`MPI_PROC_NULL`).
/// @param source Output: the rank to receive from.
/// @param dest   Output: the rank to send to.
/// @throws mpi_error if `MPI_Cart_shift` fails.
template <convertible_to_mpi_handle<MPI_Comm> Comm>
void cart_shift(Comm const& c, int direction, int disp, int& source, int& dest) {
    detail::assert_cart(mpi::experimental::handle(c));
    int err = MPI_Cart_shift(mpi::experimental::handle(c), direction, disp, &source, &dest);
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief The positions `disp` steps back (`source`) and ahead (`dest`) along dimension `direction` of the calling
/// process in the Cartesian communicator `c` (`MPI_Cart_shift`). In a non-periodic dimension, there is no neighbor
/// beyond the ends: the position stands for `MPI_PROC_NULL`.
///
/// @code
/// auto [left, right] = cart_shift(grid, 1, 1);
/// sendrecv_replace(block, right, left, grid);  // shift along the rows; skipped where there is no neighbor
/// @endcode
/// @throws mpi_error if an MPI call fails.
template <convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] cart_shift_result cart_shift(Comm const& c, int direction, int disp) {
    int source = MPI_PROC_NULL;
    int dest   = MPI_PROC_NULL;
    mpi::experimental::cart_shift(c, direction, disp, source, dest);
    cart_position const me(c);
    return cart_shift_result{
        detail::cart_position_access::shifted(me, direction, -disp, source),
        detail::cart_position_access::shifted(me, direction, disp, dest)
    };
}

// ── MPI_Cart_sub ───────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief Collective: splits the Cartesian communicator `c` into lower-dimensional grids that keep the dimensions where
/// `remain_dims` is true (`MPI_Cart_sub`).
/// @param new_comm Output: receives the grid of the calling process.
/// @throws mpi_error if `MPI_Cart_sub` fails.
template <
    convertible_to_mpi_handle<MPI_Comm>     Comm,
    flag_array                              RemainDims,
    convertible_to_mpi_handle_ptr<MPI_Comm> NewComm>
void cart_sub(Comm const& c, RemainDims const& remain_dims, NewComm& new_comm) {
    detail::assert_cart(mpi::experimental::handle(c));
    KAMPING_V2_ASSERT(
        static_cast<int>(std::ranges::size(remain_dims)) == mpi::experimental::cartdim(c),
        "cart_sub: remain_dims must have one entry per dimension"
    );
    int err = detail::with_int_array<bool>(remain_dims, [&](int const* remain_ptr) {
        return MPI_Cart_sub(mpi::experimental::handle(c), remain_ptr, mpi::experimental::handle_ptr(new_comm));
    });
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief Collective: splits the Cartesian communicator `c` into lower-dimensional grids that keep the dimensions where
/// `remain_dims` is true (`MPI_Cart_sub`), e.g. `cart_sub(grid, std::array{true, false, true})`.
/// @return The grid of the calling process, owned.
/// @throws mpi_error if `MPI_Cart_sub` fails.
template <convertible_to_mpi_handle<MPI_Comm> Comm, flag_array RemainDims>
[[nodiscard]] comm cart_sub(Comm const& c, RemainDims const& remain_dims) {
    MPI_Comm out = MPI_COMM_NULL;
    mpi::experimental::cart_sub(c, remain_dims, out);
    return comm::from_native(out);
}

/// @brief Collective: splits the Cartesian communicator `c` into lower-dimensional grids that keep the dimensions where
/// `remain_dims` is true (`MPI_Cart_sub`), e.g. `cart_sub(grid, {true, false, true})`.
/// @return The grid of the calling process, owned.
/// @throws mpi_error if `MPI_Cart_sub` fails.
template <convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] comm cart_sub(Comm const& c, std::initializer_list<bool> remain_dims) {
    return mpi::experimental::cart_sub(c, std::span<bool const>(remain_dims.begin(), remain_dims.size()));
}

/// @brief Collective: splits the Cartesian communicator `c` into one-dimensional grids along dimension `dim`, e.g. the
/// rows of a two-dimensional grid with `cart_sub(grid, 1)` (`MPI_Cart_sub`).
/// @return The grid of the calling process, owned.
/// @throws mpi_error if `MPI_Cart_sub` fails.
template <convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] comm cart_sub(Comm const& c, int dim) {
    int const ndims = mpi::experimental::cartdim(c);
    KAMPING_V2_ASSERT(dim >= 0 && dim < ndims, "cart_sub: dim is not a dimension of the grid");
    std::vector<int> remain_dims(static_cast<std::size_t>(ndims), 0);
    remain_dims[static_cast<std::size_t>(dim)] = 1;
    return mpi::experimental::cart_sub(c, remain_dims);
}

// ── comm_accessors members (declared in mpi/comm.hpp) ──────────────────────────────────────────────────────────────

template <typename Derived>
inline mpi::experimental::cart_position comm_accessors<Derived>::cart_position() const {
    return mpi::experimental::cart_position(underlying());
}

template <typename Derived>
inline mpi::experimental::cart_position comm_accessors<Derived>::cart_position(int rank) const {
    return mpi::experimental::cart_position(underlying(), rank);
}

template <typename Derived>
inline int comm_accessors<Derived>::cartdim() const {
    return mpi::experimental::cartdim(underlying());
}

template <typename Derived>
inline std::vector<int> comm_accessors<Derived>::cart_coords() const {
    return mpi::experimental::cart_coords(underlying());
}

template <typename Derived>
inline std::vector<int> comm_accessors<Derived>::cart_coords(int rank) const {
    return mpi::experimental::cart_coords(underlying(), rank);
}

template <typename Derived>
template <std::size_t N>
std::array<int, N> comm_accessors<Derived>::cart_coords() const {
    return mpi::experimental::cart_coords<N>(underlying());
}

template <typename Derived>
template <std::size_t N>
std::array<int, N> comm_accessors<Derived>::cart_coords(int rank) const {
    return mpi::experimental::cart_coords<N>(underlying(), rank);
}

template <typename Derived>
inline int comm_accessors<Derived>::cart_rank(std::span<int const> coords) const {
    return mpi::experimental::cart_rank(underlying(), coords);
}

template <typename Derived>
inline int comm_accessors<Derived>::cart_rank(std::initializer_list<int> coords) const {
    return mpi::experimental::cart_rank(underlying(), coords);
}

template <typename Derived>
inline cart_shift_result comm_accessors<Derived>::cart_shift(int direction, int disp) const {
    return mpi::experimental::cart_shift(underlying(), direction, disp);
}

template <typename Derived>
template <flag_array RemainDims>
comm comm_accessors<Derived>::cart_sub(RemainDims const& remain_dims) const {
    return mpi::experimental::cart_sub(underlying(), remain_dims);
}

template <typename Derived>
inline comm comm_accessors<Derived>::cart_sub(std::initializer_list<bool> remain_dims) const {
    return mpi::experimental::cart_sub(underlying(), remain_dims);
}

template <typename Derived>
inline comm comm_accessors<Derived>::cart_sub(int dim) const {
    return mpi::experimental::cart_sub(underlying(), dim);
}

} // namespace mpi::experimental
