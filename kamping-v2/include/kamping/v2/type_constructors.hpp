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
#include "mpi/range_concepts.hpp"
#include "mpi/rank.hpp"
#include "mpi/type_constructors.hpp"

/// @file
/// The MPI datatype constructors in `kamping::v2`: the core wrappers, plus forms that take one descriptor per block or
/// dimension instead of MPI's parallel arrays.

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

/// @brief A block of `type_indexed`: `length` elements, `displacement` elements from the start.
struct block {
    int displacement; ///< Offset in elements of the base datatype.
    int length;       ///< Number of elements.
};

/// @brief A block of `type_create_hindexed`: `length` elements, `displacement` bytes from the start.
struct hblock {
    MPI_Aint displacement; ///< Offset in bytes.
    int      length;       ///< Number of elements.
};

/// @brief A dimension of `type_create_subarray`: the `subsize` elements from index `start` of `size` elements.
struct subarray_dim {
    int size;    ///< Elements of the full array.
    int start;   ///< First index of the subarray.
    int subsize; ///< Elements of the subarray.
};

/// @brief How a dimension of `type_create_darray` is distributed.
enum class distribution : int {
    block  = MPI_DISTRIBUTE_BLOCK,  ///< One contiguous block per process.
    cyclic = MPI_DISTRIBUTE_CYCLIC, ///< Blocks dealt out round-robin.
    none   = MPI_DISTRIBUTE_NONE    ///< Not distributed.
};

/// @brief A dimension of `type_create_darray`: `global_size` elements, distributed as `distrib` over `procs`
/// processes.
struct darray_dim {
    int                global_size;                     ///< Elements of the global array.
    distribution       distrib    = distribution::none; ///< Distribution; `none` requires `procs == 1`.
    int                procs      = 1;                  ///< Processes in this dimension.
    std::optional<int> block_size = std::nullopt;       ///< Block size; MPI's default if empty.
};

namespace detail {
/// Calls `f` with the values of `r` as a `std::span<T const>`. A contiguous range of `T` is passed as it is; any other
/// range is copied, checking that each value fits into `T`.
template <typename T, mpi::experimental::sized_integer_range R, typename F>
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

/// @brief `type_indexed` with one `block` per block: `type_indexed({{0, 3}, {4, 2}}, MPI_INT)`.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_indexed` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_indexed(std::span<block const> blocks, Datatype const& oldtype) {
    detail::block_arrays<block> const arrays(blocks);
    return mpi::experimental::type_indexed(arrays.lengths, arrays.displacements, oldtype);
}

/// @brief `type_indexed` with one `block` per block: `type_indexed({{0, 3}, {4, 2}}, MPI_INT)`.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_indexed` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_indexed(std::initializer_list<block> blocks, Datatype const& oldtype) {
    return kamping::v2::type_indexed(std::span<block const>(blocks.begin(), blocks.size()), oldtype);
}

/// @brief `type_create_hindexed` with one `hblock` per block.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_hindexed` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_hindexed(std::span<hblock const> blocks, Datatype const& oldtype) {
    detail::block_arrays<hblock> const arrays(blocks);
    return mpi::experimental::type_create_hindexed(arrays.lengths, arrays.displacements, oldtype);
}

/// @brief `type_create_hindexed` with one `hblock` per block.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_hindexed` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_hindexed(std::initializer_list<hblock> blocks, Datatype const& oldtype) {
    return kamping::v2::type_create_hindexed(std::span<hblock const>(blocks.begin(), blocks.size()), oldtype);
}

// ── MPI_Type_create_indexed_block / MPI_Type_create_hindexed_block ─────────────────────────────────────────────────

/// @brief `type_create_indexed_block` with displacements from any sized integer range; without `blocklength`, one
/// element per index.
///
/// A range that is not a contiguous range of `int` is copied into a temporary array, checking that each value fits.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_indexed_block` fails.
template <mpi::experimental::sized_integer_range R, mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_indexed_block(int blocklength, R&& displacements, Datatype const& oldtype) {
    return detail::with_span_of<int>(std::forward<R>(displacements), [&](std::span<int const> displs) {
        return mpi::experimental::type_create_indexed_block(blocklength, displs, oldtype);
    });
}

/// @brief `type_create_indexed_block` with displacements from any sized integer range; without `blocklength`, one
/// element per index.
///
/// A range that is not a contiguous range of `int` is copied into a temporary array, checking that each value fits.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_indexed_block` fails.
template <mpi::experimental::sized_integer_range R, mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype type_create_indexed_block(R&& indices, Datatype const& oldtype) {
    return kamping::v2::type_create_indexed_block(1, std::forward<R>(indices), oldtype);
}

/// @brief `type_create_hindexed_block` with byte displacements from any sized integer range; without
/// `blocklength`, one element per offset.
///
/// A range that is not a contiguous range of `MPI_Aint` is copied into a temporary array, checking that each value
/// fits.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_hindexed_block` fails.
template <mpi::experimental::sized_integer_range R, mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_hindexed_block(int blocklength, R&& displacements, Datatype const& oldtype) {
    return detail::with_span_of<MPI_Aint>(std::forward<R>(displacements), [&](std::span<MPI_Aint const> displs) {
        return mpi::experimental::type_create_hindexed_block(blocklength, displs, oldtype);
    });
}

/// @brief `type_create_hindexed_block` with byte displacements from any sized integer range; without
/// `blocklength`, one element per offset.
///
/// A range that is not a contiguous range of `MPI_Aint` is copied into a temporary array, checking that each value
/// fits.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_hindexed_block` fails.
template <mpi::experimental::sized_integer_range R, mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype type_create_hindexed_block(R&& offsets, Datatype const& oldtype) {
    return kamping::v2::type_create_hindexed_block(1, std::forward<R>(offsets), oldtype);
}

// ── MPI_Type_create_subarray ───────────────────────────────────────────────────────────────────────────────────────

/// @brief `type_create_subarray` with one `subarray_dim` per dimension; `order` is `array_order::c` if left out.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_subarray` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_subarray(std::span<subarray_dim const> dims, array_order order, Datatype const& oldtype) {
    detail::subarray_arrays const arrays(dims);
    return mpi::experimental::type_create_subarray(arrays.sizes, arrays.subsizes, arrays.starts, order, oldtype);
}

/// @brief `type_create_subarray` with one `subarray_dim` per dimension; `order` is `array_order::c` if left out.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_subarray` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_subarray(std::initializer_list<subarray_dim> dims, array_order order, Datatype const& oldtype) {
    return kamping::v2::type_create_subarray(std::span<subarray_dim const>(dims.begin(), dims.size()), order, oldtype);
}

/// @brief `type_create_subarray` with one `subarray_dim` per dimension; `order` is `array_order::c` if left out.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_subarray` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_subarray(std::span<subarray_dim const> dims, Datatype const& oldtype) {
    return kamping::v2::type_create_subarray(dims, array_order::c, oldtype);
}

/// @brief `type_create_subarray` with one `subarray_dim` per dimension; `order` is `array_order::c` if left out.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_subarray` fails.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_subarray(std::initializer_list<subarray_dim> dims, Datatype const& oldtype) {
    return kamping::v2::type_create_subarray(dims, array_order::c, oldtype);
}

// ── MPI_Type_create_darray ─────────────────────────────────────────────────────────────────────────────────────────

/// @brief `type_create_darray` with one `darray_dim` per dimension, for the calling process of `comm`; `order` is
/// `array_order::c` if left out.
///
/// The product of the `procs` must be the size of `comm`.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_darray` fails.
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

/// @brief `type_create_darray` with one `darray_dim` per dimension, for the calling process of `comm`; `order` is
/// `array_order::c` if left out.
///
/// The product of the `procs` must be the size of `comm`.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_darray` fails.
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

/// @brief `type_create_darray` with one `darray_dim` per dimension, for the calling process of `comm`; `order` is
/// `array_order::c` if left out.
///
/// The product of the `procs` must be the size of `comm`.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_darray` fails.
template <
    mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>     Comm>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_darray(std::span<darray_dim const> dims, Datatype const& oldtype, Comm const& comm) {
    return kamping::v2::type_create_darray(dims, array_order::c, oldtype, comm);
}

/// @brief `type_create_darray` with one `darray_dim` per dimension, for the calling process of `comm`; `order` is
/// `array_order::c` if left out.
///
/// The product of the `procs` must be the size of `comm`.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_darray` fails.
template <
    mpi::experimental::convertible_to_mpi_handle<MPI_Datatype> Datatype,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>     Comm>
[[nodiscard]] mpi::experimental::uncommitted_datatype
type_create_darray(std::initializer_list<darray_dim> dims, Datatype const& oldtype, Comm const& comm) {
    return kamping::v2::type_create_darray(dims, array_order::c, oldtype, comm);
}

} // namespace kamping::v2
