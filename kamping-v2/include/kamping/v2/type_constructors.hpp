// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/datatype.hpp"
#include "mpi/handle.hpp"
#include "mpi/rank.hpp"
#include "mpi/type_constructors.hpp"

/// @file
/// The MPI datatype constructors in `kamping::v2`: the core wrappers from `mpi/type_constructors.hpp`, plus forms
/// that describe blocks and dimensions with one descriptor each instead of MPI's parallel arrays.

namespace kamping::v2 {
using mpi::experimental::array_order;
using mpi::experimental::type_contiguous;
using mpi::experimental::type_create_darray;
using mpi::experimental::type_create_hindexed;
using mpi::experimental::type_create_hindexed_block;
using mpi::experimental::type_create_hvector;
using mpi::experimental::type_create_indexed_block;
using mpi::experimental::type_create_resized;
using mpi::experimental::type_create_struct;
using mpi::experimental::type_create_subarray;
using mpi::experimental::type_indexed;
using mpi::experimental::type_vector;

// ── Descriptors ────────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief A block of a `type_indexed` datatype: `length` copies of the base datatype, starting `displacement`
/// elements of the base datatype after the start of the datatype.
struct block {
    int displacement; ///< Offset of the block, in elements of the base datatype.
    int length;       ///< Number of elements in the block.
};

/// @brief A block of a `type_create_hindexed` datatype: `length` copies of the base datatype, starting
/// `displacement` bytes after the start of the datatype.
struct hblock {
    MPI_Aint displacement; ///< Offset of the block, in bytes.
    int      length;       ///< Number of elements in the block.
};

/// @brief One dimension of a `type_create_subarray` datatype: the `subsize` elements starting at index `start` of a
/// dimension with `size` elements.
struct subarray_dim {
    int size;    ///< Number of elements of the full array in this dimension.
    int start;   ///< Index of the first element of the subarray in this dimension.
    int subsize; ///< Number of elements of the subarray in this dimension.
};

/// @brief How a dimension of a `type_create_darray` datatype is distributed over the processes.
enum class distribution : int {
    block  = MPI_DISTRIBUTE_BLOCK,  ///< Contiguous blocks, one per process.
    cyclic = MPI_DISTRIBUTE_CYCLIC, ///< Blocks dealt out to the processes round-robin.
    none   = MPI_DISTRIBUTE_NONE    ///< Not distributed: every process holds the whole dimension.
};

/// @brief One dimension of a `type_create_darray` datatype: `global_size` elements, distributed as `distrib` over
/// `procs` processes.
///
/// Without a `block_size`, MPI picks the default: for `distribution::block`, the global size divided by `procs`,
/// rounded up; for `distribution::cyclic`, one element. A dimension that is not distributed has `procs == 1` and no
/// block size.
struct darray_dim {
    int                global_size;                     ///< Number of elements of the global array in this dimension.
    distribution       distrib    = distribution::none; ///< How the dimension is distributed.
    int                procs      = 1;                  ///< Number of processes this dimension is distributed over.
    std::optional<int> block_size = std::nullopt;       ///< Block size of the distribution; MPI's default if empty.
};

namespace detail {
/// A sized range whose elements are integers, e.g. a `std::vector<std::size_t>` of indices.
template <typename R>
concept integer_range =
    std::ranges::input_range<R> && std::ranges::sized_range<R> && std::integral<std::ranges::range_value_t<R>>
    && !std::same_as<std::ranges::range_value_t<R>, bool>;

/// Calls `f` with the values of `r` as a `std::span<T const>`. A contiguous range of `T` is passed as it is; any other
/// range is copied, checking that each value fits into `T`.
template <typename T, integer_range R, typename F>
decltype(auto) with_span_of(R&& r, F&& f) {
    if constexpr (std::ranges::contiguous_range<R> && std::same_as<std::ranges::range_value_t<R>, T>) {
        return std::forward<F>(f)(std::span<T const>(std::ranges::data(r), std::ranges::size(r)));
    } else {
        std::vector<T> values;
        values.reserve(static_cast<std::size_t>(std::ranges::size(r)));
        for (auto const v: r) {
            KAMPING_V2_ASSERT(std::in_range<T>(v), "displacement does not fit into the type MPI expects");
            values.push_back(static_cast<T>(v));
        }
        return std::forward<F>(f)(std::span<T const>(values));
    }
}

/// The parallel arrays MPI expects for a sequence of blocks.
template <typename Block>
struct block_arrays {
    std::vector<int>                                          lengths;
    std::vector<decltype(std::declval<Block>().displacement)> displacements;

    explicit block_arrays(std::span<Block const> blocks) {
        lengths.reserve(blocks.size());
        displacements.reserve(blocks.size());
        for (auto const& b: blocks) {
            lengths.push_back(b.length);
            displacements.push_back(b.displacement);
        }
    }
};

/// The parallel arrays MPI expects for the dimensions of a subarray.
struct subarray_arrays {
    std::vector<int> sizes, subsizes, starts;

    explicit subarray_arrays(std::span<subarray_dim const> dims) {
        for (auto const& d: dims) {
            sizes.push_back(d.size);
            subsizes.push_back(d.subsize);
            starts.push_back(d.start);
        }
    }
};

/// The parallel arrays MPI expects for the dimensions of a distributed array.
struct darray_arrays {
    std::vector<int> gsizes, distribs, dargs, psizes;

    explicit darray_arrays(std::span<darray_dim const> dims) {
        for (auto const& d: dims) {
            KAMPING_V2_ASSERT(
                d.distrib != distribution::none || (d.procs == 1 && !d.block_size.has_value()),
                "type_create_darray: a dimension that is not distributed has procs == 1 and no block size"
            );
            gsizes.push_back(d.global_size);
            distribs.push_back(static_cast<int>(d.distrib));
            dargs.push_back(d.block_size.value_or(MPI_DISTRIBUTE_DFLT_DARG));
            psizes.push_back(d.procs);
        }
    }
};
} // namespace detail

// ── MPI_Type_indexed / MPI_Type_create_hindexed ────────────────────────────────────────────────────────────────────

/// @brief Blocks of `oldtype` of different lengths at different offsets (`MPI_Type_indexed`).
///
/// Each block gives its offset and length in elements of `oldtype`:
///
/// @code
/// // The upper triangle of a 3 x 3 row-major matrix: row i holds 3 - i elements, starting at column i.
/// auto const upper = v2::type_indexed({{0, 3}, {4, 2}, {8, 1}}, MPI_DOUBLE).commit();
/// @endcode
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_indexed` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_indexed(std::span<block const> blocks, Datatype const& oldtype) {
    detail::block_arrays<block> const arrays(blocks);
    return mpi::experimental::type_indexed(arrays.lengths, arrays.displacements, oldtype);
}

/// @brief Blocks of `oldtype` of different lengths at different offsets (`MPI_Type_indexed`).
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_indexed` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_indexed(std::initializer_list<block> blocks, Datatype const& oldtype) {
    return kamping::v2::type_indexed(std::span<block const>(blocks.begin(), blocks.size()), oldtype);
}

/// @brief Like `type_indexed`, but the offsets of the blocks are given in bytes (`MPI_Type_create_hindexed`).
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_hindexed` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_hindexed(std::span<hblock const> blocks, Datatype const& oldtype) {
    detail::block_arrays<hblock> const arrays(blocks);
    return mpi::experimental::type_create_hindexed(arrays.lengths, arrays.displacements, oldtype);
}

/// @brief Like `type_indexed`, but the offsets of the blocks are given in bytes (`MPI_Type_create_hindexed`).
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_hindexed` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_hindexed(std::initializer_list<hblock> blocks, Datatype const& oldtype) {
    return kamping::v2::type_create_hindexed(std::span<hblock const>(blocks.begin(), blocks.size()), oldtype);
}

// ── MPI_Type_create_indexed_block / MPI_Type_create_hindexed_block ─────────────────────────────────────────────────

/// @brief Blocks of `blocklength` copies of `oldtype` each at different offsets (`MPI_Type_create_indexed_block`).
///
/// Block `i` starts `displacements[i]` elements of `oldtype` after the start of the datatype. `displacements` can be
/// any sized range of integers; each must fit into an `int`.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_indexed_block` fails.
template <detail::integer_range R, mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_indexed_block(int blocklength, R&& displacements, Datatype const& oldtype) {
    return detail::with_span_of<int>(std::forward<R>(displacements), [&](std::span<int const> displs) {
        return mpi::experimental::type_create_indexed_block(blocklength, displs, oldtype);
    });
}

/// @brief The elements of an array of `oldtype` at the given indices, in that order
/// (`MPI_Type_create_indexed_block` with blocks of one element).
///
/// `indices` can be any sized range of integers; each must fit into an `int`.
///
/// @code
/// std::vector<std::size_t> const selected{0, 5, 7};  // the elements 0, 5 and 7 of an array
/// auto const                     gather = v2::type_create_indexed_block(selected, MPI_INT).commit();
/// @endcode
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_indexed_block` fails.
template <detail::integer_range R, mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype type_create_indexed_block(R&& indices, Datatype const& oldtype) {
    return kamping::v2::type_create_indexed_block(1, std::forward<R>(indices), oldtype);
}

/// @brief Like `type_create_indexed_block`, but `displacements` are given in bytes
/// (`MPI_Type_create_hindexed_block`).
///
/// `displacements` can be any sized range of integers; each must fit into an `MPI_Aint`.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_hindexed_block` fails.
template <detail::integer_range R, mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_hindexed_block(int blocklength, R&& displacements, Datatype const& oldtype) {
    return detail::with_span_of<MPI_Aint>(std::forward<R>(displacements), [&](std::span<MPI_Aint const> displs) {
        return mpi::experimental::type_create_hindexed_block(blocklength, displs, oldtype);
    });
}

/// @brief One element of `oldtype` at each of the given byte offsets, in that order
/// (`MPI_Type_create_hindexed_block` with blocks of one element).
///
/// `offsets` can be any sized range of integers; each must fit into an `MPI_Aint`.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_hindexed_block` fails.
template <detail::integer_range R, mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype type_create_hindexed_block(R&& offsets, Datatype const& oldtype) {
    return kamping::v2::type_create_hindexed_block(1, std::forward<R>(offsets), oldtype);
}

// ── MPI_Type_create_subarray ───────────────────────────────────────────────────────────────────────────────────────

/// @brief The subarray of a multi-dimensional array of `oldtype` described by one `subarray_dim` per dimension
/// (`MPI_Type_create_subarray`).
///
/// The array is stored in `order`, row-major (`array_order::c`) if left out. The extent of the new datatype is that
/// of the full array, so a buffer holding the full array is sent or received with a count of one.
///
/// @code
/// // A rows x cols matrix without its boundary.
/// auto interior = v2::type_create_subarray({{rows, 1, rows - 2}, {cols, 1, cols - 2}}, MPI_DOUBLE).commit();
/// @endcode
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_subarray` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_subarray(std::span<subarray_dim const> dims, array_order order, Datatype const& oldtype) {
    detail::subarray_arrays const arrays(dims);
    return mpi::experimental::type_create_subarray(arrays.sizes, arrays.subsizes, arrays.starts, order, oldtype);
}

/// @brief The subarray of a multi-dimensional array of `oldtype` described by one `subarray_dim` per dimension
/// (`MPI_Type_create_subarray`).
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_subarray` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_subarray(std::initializer_list<subarray_dim> dims, array_order order, Datatype const& oldtype) {
    return kamping::v2::type_create_subarray(std::span<subarray_dim const>(dims.begin(), dims.size()), order, oldtype);
}

/// @brief The subarray of a multi-dimensional array of `oldtype` described by one `subarray_dim` per dimension
/// (`MPI_Type_create_subarray`).
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_subarray` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_subarray(std::span<subarray_dim const> dims, Datatype const& oldtype) {
    return kamping::v2::type_create_subarray(dims, array_order::c, oldtype);
}

/// @brief The subarray of a multi-dimensional array of `oldtype` described by one `subarray_dim` per dimension
/// (`MPI_Type_create_subarray`).
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_subarray` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_subarray(std::initializer_list<subarray_dim> dims, Datatype const& oldtype) {
    return kamping::v2::type_create_subarray(dims, array_order::c, oldtype);
}

// ── MPI_Type_create_darray ─────────────────────────────────────────────────────────────────────────────────────────

/// @brief The part of a distributed multi-dimensional array of `oldtype` that the calling process holds
/// (`MPI_Type_create_darray`).
///
/// The array has one `darray_dim` per dimension and is stored in `order`, row-major (`array_order::c`) if left out.
/// The processes of `comm` form a grid with `dims[i].procs` processes in dimension `i`, in row-major order; the
/// product of the `procs` must be the size of `comm`.
///
/// @code
/// // An n x n matrix in blocks on a p_rows x p_cols process grid.
/// using v2::distribution;
/// auto const blk =
///     v2::type_create_darray({{n, distribution::block, p_rows}, {n, distribution::block, p_cols}}, MPI_DOUBLE, comm)
///         .commit();
/// @endcode
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_darray` or querying the rank or size of `comm` fails.
template <
    mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>     Comm>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_darray(std::span<darray_dim const> dims, array_order order, Datatype const& oldtype, Comm const& comm) {
    MPI_Comm const              raw_comm = mpi::experimental::handle(comm);
    detail::darray_arrays const arrays(dims);
    return mpi::experimental::type_create_darray(
        mpi::experimental::detail::comm_size(raw_comm),
        mpi::experimental::detail::comm_rank(raw_comm),
        arrays.gsizes,
        arrays.distribs,
        arrays.dargs,
        arrays.psizes,
        order,
        oldtype
    );
}

/// @brief The part of a distributed multi-dimensional array of `oldtype` that the calling process holds
/// (`MPI_Type_create_darray`).
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_darray` or querying the rank or size of `comm` fails.
template <
    mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>     Comm>
[[nodiscard]] mpi::experimental::uncommitted_datatype type_create_darray(
    std::initializer_list<darray_dim> dims, array_order order, Datatype const& oldtype, Comm const& comm
) {
    return kamping::v2::type_create_darray(
        std::span<darray_dim const>(dims.begin(), dims.size()),
        order,
        oldtype,
        comm
    );
}

/// @brief The part of a distributed multi-dimensional array of `oldtype` that the calling process holds
/// (`MPI_Type_create_darray`).
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_darray` or querying the rank or size of `comm` fails.
template <
    mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>     Comm>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_darray(std::span<darray_dim const> dims, Datatype const& oldtype, Comm const& comm) {
    return kamping::v2::type_create_darray(dims, array_order::c, oldtype, comm);
}

/// @brief The part of a distributed multi-dimensional array of `oldtype` that the calling process holds
/// (`MPI_Type_create_darray`).
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_darray` or querying the rank or size of `comm` fails.
template <
    mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>     Comm>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_darray(std::initializer_list<darray_dim> dims, Datatype const& oldtype, Comm const& comm) {
    return kamping::v2::type_create_darray(dims, array_order::c, oldtype, comm);
}

} // namespace kamping::v2
