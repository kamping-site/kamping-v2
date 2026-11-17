// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/datatype.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"
#include "mpi/int_array.hpp"
#include "mpi/range_concepts.hpp"

/// @file
/// Wrappers for the MPI datatype constructors (`MPI_Type_contiguous`, `MPI_Type_vector`, ...).
///
/// Each constructor comes in two forms:
/// - The full form mirrors the MPI call and writes the new datatype into its last argument, any
///   `convertible_to_mpi_handle_ptr<MPI_Datatype>`: a raw `MPI_Datatype` or an empty `uncommitted_datatype`.
/// - The short form leaves out that argument and returns the new datatype as an owning `uncommitted_datatype`.
///
/// The new datatype is never committed; `std::move(u).commit()` turns an `uncommitted_datatype` into a `datatype` that
/// can be used as the datatype of a buffer. Base datatypes are taken as any `convertible_to_mpi_handle<MPI_Datatype>`:
/// a `datatype`, an `uncommitted_datatype` (MPI allows uncommitted base datatypes), a `datatype_view` or a raw
/// `MPI_Datatype`. A base datatype may be freed once the call returns, so a temporary `uncommitted_datatype` can be
/// passed directly. Array arguments are contiguous ranges; MPI's count is their length, and the lengths of the arrays
/// of one call are checked against each other.

namespace mpi::experimental {

/// @brief Storage order of a multi-dimensional array for `type_create_subarray` and `type_create_darray`.
enum class array_order : int {
    c       = MPI_ORDER_C,      ///< Row-major: the last dimension varies fastest.
    fortran = MPI_ORDER_FORTRAN ///< Column-major: the first dimension varies fastest.
};

/// @brief How a dimension of `type_create_darray` is distributed (`MPI_DISTRIBUTE_*`).
enum class distribution : int {
    block  = MPI_DISTRIBUTE_BLOCK,  ///< One contiguous block per process.
    cyclic = MPI_DISTRIBUTE_CYCLIC, ///< Blocks dealt out round-robin.
    none   = MPI_DISTRIBUTE_NONE    ///< Not distributed.
};

/// @brief The distributions of `type_create_darray`: a contiguous range of `int` (`MPI_DISTRIBUTE_*`), passed to MPI as
/// it is, or a sized range of `distribution`.
template <typename R>
concept distribution_range = int_array_of<R, distribution>;

// ── MPI_Type_contiguous ────────────────────────────────────────────────────────────────────────────────────────────

/// @brief `count` copies of `oldtype`, laid out contiguously (`MPI_Type_contiguous`).
/// @param newtype Output: receives the new, uncommitted datatype.
/// @throws mpi_error if `MPI_Type_contiguous` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype, convertible_to_mpi_handle_ptr<MPI_Datatype> NewDatatype>
void type_contiguous(int count, Datatype const& oldtype, NewDatatype& newtype) {
    int err = MPI_Type_contiguous(count, mpi::experimental::handle(oldtype), mpi::experimental::handle_ptr(newtype));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief `count` copies of `oldtype`, laid out contiguously (`MPI_Type_contiguous`).
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_contiguous` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] uncommitted_datatype type_contiguous(int count, Datatype const& oldtype) {
    uncommitted_datatype newtype;
    type_contiguous(count, oldtype, newtype);
    return newtype;
}

// ── MPI_Type_vector / MPI_Type_create_hvector ──────────────────────────────────────────────────────────────────────

/// @brief `count` blocks of `blocklength` copies of `oldtype` each, whose starts are `stride` elements of `oldtype`
/// apart (`MPI_Type_vector`).
/// @param newtype Output: receives the new, uncommitted datatype.
/// @throws mpi_error if `MPI_Type_vector` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype, convertible_to_mpi_handle_ptr<MPI_Datatype> NewDatatype>
void type_vector(int count, int blocklength, int stride, Datatype const& oldtype, NewDatatype& newtype) {
    int err = MPI_Type_vector(
        count,
        blocklength,
        stride,
        mpi::experimental::handle(oldtype),
        mpi::experimental::handle_ptr(newtype)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief `count` blocks of `blocklength` copies of `oldtype` each, whose starts are `stride` elements of `oldtype`
/// apart (`MPI_Type_vector`).
///
/// @code
/// auto const column = type_vector(n, 1, n, MPI_DOUBLE);  // one column of an n x n row-major matrix
/// @endcode
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_vector` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] uncommitted_datatype type_vector(int count, int blocklength, int stride, Datatype const& oldtype) {
    uncommitted_datatype newtype;
    type_vector(count, blocklength, stride, oldtype, newtype);
    return newtype;
}

/// @brief Like `type_vector`, but `stride` is given in bytes (`MPI_Type_create_hvector`).
/// @param newtype Output: receives the new, uncommitted datatype.
/// @throws mpi_error if `MPI_Type_create_hvector` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype, convertible_to_mpi_handle_ptr<MPI_Datatype> NewDatatype>
void type_create_hvector(int count, int blocklength, MPI_Aint stride, Datatype const& oldtype, NewDatatype& newtype) {
    int err = MPI_Type_create_hvector(
        count,
        blocklength,
        stride,
        mpi::experimental::handle(oldtype),
        mpi::experimental::handle_ptr(newtype)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief Like `type_vector`, but `stride` is given in bytes (`MPI_Type_create_hvector`).
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_hvector` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] uncommitted_datatype
type_create_hvector(int count, int blocklength, MPI_Aint stride, Datatype const& oldtype) {
    uncommitted_datatype newtype;
    type_create_hvector(count, blocklength, stride, oldtype, newtype);
    return newtype;
}

// ── MPI_Type_indexed / MPI_Type_create_hindexed ────────────────────────────────────────────────────────────────────

/// @brief Blocks of `oldtype` of different lengths at different offsets (`MPI_Type_indexed`).
///
/// Block `i` holds `blocklengths[i]` copies of `oldtype` and starts `displacements[i]` elements of `oldtype` after the
/// start of the datatype. Both arrays must have the same length, the number of blocks.
/// @param newtype Output: receives the new, uncommitted datatype.
/// @throws mpi_error if `MPI_Type_indexed` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype, convertible_to_mpi_handle_ptr<MPI_Datatype> NewDatatype>
void type_indexed(
    std::span<int const> blocklengths, std::span<int const> displacements, Datatype const& oldtype, NewDatatype& newtype
) {
    KAMPING_V2_ASSERT(
        blocklengths.size() == displacements.size(),
        "type_indexed: blocklengths and displacements must have the same length"
    );
    int err = MPI_Type_indexed(
        static_cast<int>(blocklengths.size()),
        blocklengths.data(),
        displacements.data(),
        mpi::experimental::handle(oldtype),
        mpi::experimental::handle_ptr(newtype)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief Blocks of `oldtype` of different lengths at different offsets (`MPI_Type_indexed`).
///
/// Block `i` holds `blocklengths[i]` copies of `oldtype` and starts `displacements[i]` elements of `oldtype` after the
/// start of the datatype. Both arrays must have the same length, the number of blocks.
///
/// @code
/// // The upper triangle of an n x n row-major matrix: row i holds n - i elements, starting at column i.
/// std::vector<int> blocklengths(n), displacements(n);
/// for (int i = 0; i < n; ++i) {
///     blocklengths[i]  = n - i;
///     displacements[i] = i * n + i;
/// }
/// auto const upper = type_indexed(blocklengths, displacements, MPI_DOUBLE).commit();
/// @endcode
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_indexed` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] uncommitted_datatype
type_indexed(std::span<int const> blocklengths, std::span<int const> displacements, Datatype const& oldtype) {
    uncommitted_datatype newtype;
    type_indexed(blocklengths, displacements, oldtype, newtype);
    return newtype;
}

/// @brief Like `type_indexed`, but `displacements` are given in bytes (`MPI_Type_create_hindexed`).
/// @param newtype Output: receives the new, uncommitted datatype.
/// @throws mpi_error if `MPI_Type_create_hindexed` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype, convertible_to_mpi_handle_ptr<MPI_Datatype> NewDatatype>
void type_create_hindexed(
    std::span<int const>      blocklengths,
    std::span<MPI_Aint const> displacements,
    Datatype const&           oldtype,
    NewDatatype&              newtype
) {
    KAMPING_V2_ASSERT(
        blocklengths.size() == displacements.size(),
        "type_create_hindexed: blocklengths and displacements must have the same length"
    );
    int err = MPI_Type_create_hindexed(
        static_cast<int>(blocklengths.size()),
        blocklengths.data(),
        displacements.data(),
        mpi::experimental::handle(oldtype),
        mpi::experimental::handle_ptr(newtype)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief Like `type_indexed`, but `displacements` are given in bytes (`MPI_Type_create_hindexed`).
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_hindexed` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] uncommitted_datatype type_create_hindexed(
    std::span<int const> blocklengths, std::span<MPI_Aint const> displacements, Datatype const& oldtype
) {
    uncommitted_datatype newtype;
    type_create_hindexed(blocklengths, displacements, oldtype, newtype);
    return newtype;
}

// ── MPI_Type_create_indexed_block / MPI_Type_create_hindexed_block ─────────────────────────────────────────────────

/// @brief Blocks of `blocklength` copies of `oldtype` each at different offsets (`MPI_Type_create_indexed_block`).
///
/// Block `i` starts `displacements[i]` elements of `oldtype` after the start of the datatype; the length of
/// `displacements` is the number of blocks.
/// @param newtype Output: receives the new, uncommitted datatype.
/// @throws mpi_error if `MPI_Type_create_indexed_block` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype, convertible_to_mpi_handle_ptr<MPI_Datatype> NewDatatype>
void type_create_indexed_block(
    int blocklength, std::span<int const> displacements, Datatype const& oldtype, NewDatatype& newtype
) {
    int err = MPI_Type_create_indexed_block(
        static_cast<int>(displacements.size()),
        blocklength,
        displacements.data(),
        mpi::experimental::handle(oldtype),
        mpi::experimental::handle_ptr(newtype)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief Blocks of `blocklength` copies of `oldtype` each at different offsets (`MPI_Type_create_indexed_block`).
///
/// Block `i` starts `displacements[i]` elements of `oldtype` after the start of the datatype; the length of
/// `displacements` is the number of blocks.
///
/// @code
/// std::array const selected{0, 5, 7};  // the elements 0, 5 and 7 of an array
/// auto const       gather = type_create_indexed_block(1, selected, MPI_INT).commit();
/// @endcode
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_indexed_block` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] uncommitted_datatype
type_create_indexed_block(int blocklength, std::span<int const> displacements, Datatype const& oldtype) {
    uncommitted_datatype newtype;
    type_create_indexed_block(blocklength, displacements, oldtype, newtype);
    return newtype;
}

/// @brief Like `type_create_indexed_block`, but `displacements` are given in bytes
/// (`MPI_Type_create_hindexed_block`).
/// @param newtype Output: receives the new, uncommitted datatype.
/// @throws mpi_error if `MPI_Type_create_hindexed_block` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype, convertible_to_mpi_handle_ptr<MPI_Datatype> NewDatatype>
void type_create_hindexed_block(
    int blocklength, std::span<MPI_Aint const> displacements, Datatype const& oldtype, NewDatatype& newtype
) {
    int err = MPI_Type_create_hindexed_block(
        static_cast<int>(displacements.size()),
        blocklength,
        displacements.data(),
        mpi::experimental::handle(oldtype),
        mpi::experimental::handle_ptr(newtype)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief Like `type_create_indexed_block`, but `displacements` are given in bytes
/// (`MPI_Type_create_hindexed_block`).
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_hindexed_block` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] uncommitted_datatype
type_create_hindexed_block(int blocklength, std::span<MPI_Aint const> displacements, Datatype const& oldtype) {
    uncommitted_datatype newtype;
    type_create_hindexed_block(blocklength, displacements, oldtype, newtype);
    return newtype;
}

// ── MPI_Type_create_struct ─────────────────────────────────────────────────────────────────────────────────────────

/// @brief A datatype made of blocks of different datatypes (`MPI_Type_create_struct`).
///
/// Block `i` holds `blocklengths[i]` copies of `types[i]` and starts `displacements[i]` bytes after the start of the
/// datatype. The three arrays must have the same length, the number of blocks.
/// @param newtype Output: receives the new, uncommitted datatype.
/// @throws mpi_error if `MPI_Type_create_struct` fails.
template <convertible_to_mpi_handle_ptr<MPI_Datatype> NewDatatype>
void type_create_struct(
    std::span<int const>          blocklengths,
    std::span<MPI_Aint const>     displacements,
    std::span<MPI_Datatype const> types,
    NewDatatype&                  newtype
) {
    KAMPING_V2_ASSERT(
        blocklengths.size() == displacements.size() && blocklengths.size() == types.size(),
        "type_create_struct: blocklengths, displacements and types must have the same length"
    );
    int err = MPI_Type_create_struct(
        static_cast<int>(blocklengths.size()),
        blocklengths.data(),
        displacements.data(),
        types.data(),
        mpi::experimental::handle_ptr(newtype)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief A datatype made of blocks of different datatypes (`MPI_Type_create_struct`).
///
/// Block `i` holds `blocklengths[i]` copies of `types[i]` and starts `displacements[i]` bytes after the start of the
/// datatype. The three arrays must have the same length, the number of blocks. To account for trailing padding,
/// resize the result to the size of the struct:
///
/// @code
/// struct point { int id; double x; };
/// std::array const               blocklengths{1, 1};
/// std::array<MPI_Aint, 2> const  displacements{offsetof(point, id), offsetof(point, x)};
/// std::array const               types{MPI_INT, MPI_DOUBLE};
/// auto point_type =
///     type_create_resized(type_create_struct(blocklengths, displacements, types), 0, sizeof(point)).commit();
/// @endcode
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_struct` fails.
[[nodiscard]] inline uncommitted_datatype type_create_struct(
    std::span<int const> blocklengths, std::span<MPI_Aint const> displacements, std::span<MPI_Datatype const> types
) {
    uncommitted_datatype newtype;
    type_create_struct(blocklengths, displacements, types, newtype);
    return newtype;
}

// ── MPI_Type_create_subarray ───────────────────────────────────────────────────────────────────────────────────────

/// @brief The subarray of `subsizes` elements starting at `starts` of a multi-dimensional array of `oldtype` with
/// `sizes` elements per dimension (`MPI_Type_create_subarray`).
///
/// `sizes`, `subsizes` and `starts` must have the same length, the number of dimensions. The extent of the new
/// datatype is that of the full array, so a buffer holding the full array is sent or received with a count of one.
/// @param newtype Output: receives the new, uncommitted datatype.
/// @throws mpi_error if `MPI_Type_create_subarray` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype, convertible_to_mpi_handle_ptr<MPI_Datatype> NewDatatype>
void type_create_subarray(
    std::span<int const> sizes,
    std::span<int const> subsizes,
    std::span<int const> starts,
    array_order          order,
    Datatype const&      oldtype,
    NewDatatype&         newtype
) {
    KAMPING_V2_ASSERT(
        sizes.size() == subsizes.size() && sizes.size() == starts.size(),
        "type_create_subarray: sizes, subsizes and starts must have the same length"
    );
    int err = MPI_Type_create_subarray(
        static_cast<int>(sizes.size()),
        sizes.data(),
        subsizes.data(),
        starts.data(),
        static_cast<int>(order),
        mpi::experimental::handle(oldtype),
        mpi::experimental::handle_ptr(newtype)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief The subarray of `subsizes` elements starting at `starts` of a multi-dimensional array of `oldtype` with
/// `sizes` elements per dimension (`MPI_Type_create_subarray`).
///
/// `sizes`, `subsizes` and `starts` must have the same length, the number of dimensions. The extent of the new
/// datatype is that of the full array, so a buffer holding the full array is sent or received with a count of one.
///
/// @code
/// std::array const sizes{rows, cols};          // a rows x cols matrix ...
/// std::array const subsizes{rows - 2, cols - 2};
/// std::array const starts{1, 1};               // ... without its boundary
/// auto interior = type_create_subarray(sizes, subsizes, starts, array_order::c, MPI_DOUBLE).commit();
/// @endcode
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_subarray` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] uncommitted_datatype type_create_subarray(
    std::span<int const> sizes,
    std::span<int const> subsizes,
    std::span<int const> starts,
    array_order          order,
    Datatype const&      oldtype
) {
    uncommitted_datatype newtype;
    type_create_subarray(sizes, subsizes, starts, order, oldtype, newtype);
    return newtype;
}

// ── MPI_Type_create_darray ─────────────────────────────────────────────────────────────────────────────────────────

/// @brief The part that process `rank` of `size` processes holds of a distributed multi-dimensional array of
/// `oldtype` (`MPI_Type_create_darray`).
///
/// Dimension `i` has `gsizes[i]` elements and is distributed as `distribs[i]` with the distribution argument
/// `dargs[i]` (a block size or `MPI_DISTRIBUTE_DFLT_DARG`) over `psizes[i]` processes; the product of `psizes` must be
/// `size`. All four arrays must have the same length, the number of dimensions. `distribs` holds `distribution`s, or
/// `int`s (`MPI_DISTRIBUTE_*`), which are passed to MPI without a copy.
/// @param newtype Output: receives the new, uncommitted datatype.
/// @throws mpi_error if `MPI_Type_create_darray` fails.
template <
    distribution_range                          Distribs,
    convertible_to_mpi_handle<MPI_Datatype>     Datatype,
    convertible_to_mpi_handle_ptr<MPI_Datatype> NewDatatype>
void type_create_darray(
    int                  size,
    int                  rank,
    std::span<int const> gsizes,
    Distribs const&      distribs,
    std::span<int const> dargs,
    std::span<int const> psizes,
    array_order          order,
    Datatype const&      oldtype,
    NewDatatype&         newtype
) {
    KAMPING_V2_ASSERT(
        gsizes.size() == static_cast<std::size_t>(std::ranges::size(distribs)) && gsizes.size() == dargs.size()
            && gsizes.size() == psizes.size(),
        "type_create_darray: gsizes, distribs, dargs and psizes must have the same length"
    );
    int err = detail::with_int_array<distribution>(distribs, [&](int const* distribs_ptr) {
        return MPI_Type_create_darray(
            size,
            rank,
            static_cast<int>(gsizes.size()),
            gsizes.data(),
            distribs_ptr,
            dargs.data(),
            psizes.data(),
            static_cast<int>(order),
            mpi::experimental::handle(oldtype),
            mpi::experimental::handle_ptr(newtype)
        );
    });
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief The part that process `rank` of `size` processes holds of a distributed multi-dimensional array of
/// `oldtype` (`MPI_Type_create_darray`).
///
/// Dimension `i` has `gsizes[i]` elements and is distributed as `distribs[i]` with the distribution argument
/// `dargs[i]` (a block size or `MPI_DISTRIBUTE_DFLT_DARG`) over `psizes[i]` processes; the product of `psizes` must be
/// `size`. All four arrays must have the same length, the number of dimensions. `distribs` holds `distribution`s, or
/// `int`s (`MPI_DISTRIBUTE_*`), which are passed to MPI without a copy.
///
/// @code
/// std::array const gsizes{n, n};  // an n x n matrix in blocks on a p_rows x p_cols process grid
/// std::array const distribs{distribution::block, distribution::block};
/// std::array const dargs{MPI_DISTRIBUTE_DFLT_DARG, MPI_DISTRIBUTE_DFLT_DARG};
/// std::array const psizes{p_rows, p_cols};
/// auto block =
///     type_create_darray(size, rank, gsizes, distribs, dargs, psizes, array_order::c, MPI_DOUBLE).commit();
/// @endcode
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_darray` fails.
template <distribution_range Distribs, convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] uncommitted_datatype type_create_darray(
    int                  size,
    int                  rank,
    std::span<int const> gsizes,
    Distribs const&      distribs,
    std::span<int const> dargs,
    std::span<int const> psizes,
    array_order          order,
    Datatype const&      oldtype
) {
    uncommitted_datatype newtype;
    type_create_darray(size, rank, gsizes, distribs, dargs, psizes, order, oldtype, newtype);
    return newtype;
}

// ── MPI_Type_create_resized ────────────────────────────────────────────────────────────────────────────────────────

/// @brief `oldtype` with its lower bound set to `lb` and its extent to `extent` bytes (`MPI_Type_create_resized`).
/// @param newtype Output: receives the new, uncommitted datatype.
/// @throws mpi_error if `MPI_Type_create_resized` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype, convertible_to_mpi_handle_ptr<MPI_Datatype> NewDatatype>
void type_create_resized(Datatype const& oldtype, MPI_Aint lb, MPI_Aint extent, NewDatatype& newtype) {
    int err =
        MPI_Type_create_resized(mpi::experimental::handle(oldtype), lb, extent, mpi::experimental::handle_ptr(newtype));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief `oldtype` with its lower bound set to `lb` and its extent to `extent` bytes (`MPI_Type_create_resized`).
///
/// The extent is the distance between consecutive elements of the datatype. Resizing sends consecutive columns of a
/// matrix, for example, or accounts for the trailing padding of a struct:
///
/// @code
/// auto column_type = type_create_resized(type_vector(n, 1, n, MPI_DOUBLE), 0, sizeof(double)).commit();
/// @endcode
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_resized` fails.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] uncommitted_datatype type_create_resized(Datatype const& oldtype, MPI_Aint lb, MPI_Aint extent) {
    uncommitted_datatype newtype;
    type_create_resized(oldtype, lb, extent, newtype);
    return newtype;
}

} // namespace mpi::experimental
