// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <array>
#include <climits>
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/buffer.hpp"
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

/// @brief A block of `type_create_struct`: `length` elements of `type`, `displacement` bytes from the start.
struct struct_field {
    MPI_Aint                         displacement; ///< Offset in bytes.
    int                              length;       ///< Number of elements.
    mpi::experimental::datatype_view type;         ///< Datatype of the elements.
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

/// The parallel arrays MPI expects for the fields of a struct.
struct struct_arrays {
    std::vector<int>          lengths;
    std::vector<MPI_Aint>     displacements;
    std::vector<MPI_Datatype> types;

    explicit struct_arrays(std::span<struct_field const> fields) {
        for (auto const& f: fields) {
            lengths.push_back(f.length);
            displacements.push_back(f.displacement);
            types.push_back(f.type.mpi_handle());
        }
    }
};

/// The element type and length of a one-dimensional array member: a C array or a `std::array`.
template <typename M>
struct array_member_traits {};
template <typename E, std::size_t N>
struct array_member_traits<E[N]> {
    using element_type                  = E;
    static constexpr std::size_t length = N;
};
template <typename E, std::size_t N>
struct array_member_traits<std::array<E, N>> {
    using element_type                  = E;
    static constexpr std::size_t length = N;
};

/// A one-dimensional array member (C array or `std::array`), ignoring cv-qualifiers.
template <typename M>
concept array_member = requires { typename array_member_traits<std::remove_cv_t<M>>::element_type; };

/// Whether `count` elements of `dt` (one every `extent` bytes) lie within a member of `size` bytes.
inline bool fits_into_member(MPI_Datatype dt, int count, std::size_t size) {
    MPI_Aint lb = 0, extent = 0, true_lb = 0, true_extent = 0;
    MPI_Type_get_extent(dt, &lb, &extent);
    MPI_Type_get_true_extent(dt, &true_lb, &true_extent);
    return true_lb >= 0
           && true_lb + static_cast<MPI_Aint>(count - 1) * extent + true_extent <= static_cast<MPI_Aint>(size);
}
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

// ── MPI_Type_create_struct ─────────────────────────────────────────────────────────────────────────────────────────

/// @brief `type_create_struct` with one `struct_field` per block: `type_create_struct({{0, 1, MPI_INT}, {8, 2,
/// MPI_DOUBLE}})`.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_struct` fails.
[[nodiscard]] inline mpi::experimental::uncommitted_datatype type_create_struct(std::span<struct_field const> fields) {
    detail::struct_arrays const arrays(fields);
    return mpi::experimental::type_create_struct(arrays.lengths, arrays.displacements, arrays.types);
}

/// @brief `type_create_struct` with one `struct_field` per block: `type_create_struct({{0, 1, MPI_INT}, {8, 2,
/// MPI_DOUBLE}})`.
/// @return The new datatype, owned and not committed.
/// @throws mpi_error if `MPI_Type_create_struct` fails.
[[nodiscard]] inline mpi::experimental::uncommitted_datatype
type_create_struct(std::initializer_list<struct_field> fields) {
    return kamping::v2::type_create_struct(std::span<struct_field const>(fields.begin(), fields.size()));
}

/// @brief Builds the datatype of a class `T` from its members, one field at a time.
///
/// For classes that `struct_type` cannot reflect (constructors, private members, base classes) or of which only some
/// members are sent. The datatype of a builtin member is deduced; any other member is given its datatype explicitly.
/// `build()` returns the datatype, resized to `sizeof(T)` so that arrays of `T` work.
///
/// @code
/// struct particle {
///     int                   id;
///     std::array<double, 3> pos;
///     cell                  where;
///     std::string           name;  // not sent
/// };
/// auto const particle_type = v2::struct_builder<particle>{}
///                                .field(&particle::id)
///                                .array_field(&particle::pos)
///                                .field(&particle::where, cell_type)
///                                .build()
///                                .commit();
/// @endcode
///
/// Field offsets are taken from an object of `T`: one the builder default-constructs, or a sample object passed to the
/// constructor, which must outlive the calls that add fields. A datatype passed as a temporary `uncommitted_datatype`
/// or `datatype` is owned by the builder and freed with it; other datatypes must stay valid until `build()`. Move-only.
template <typename T>
class struct_builder {
public:
    /// @brief A builder that takes field offsets from a default-constructed `T`.
    struct_builder()
        requires std::default_initializable<T>
        : _owned_sample(std::make_unique<T>()),
          _sample(_owned_sample.get()) {}

    /// @brief A builder that takes field offsets from `sample`, which must outlive the calls that add fields.
    explicit struct_builder(T const& sample) : _sample(std::addressof(sample)) {}

    /// @brief Adds a member of builtin type `M`, e.g. `int` or `double`, as one element of its predefined datatype.
    template <mpi::experimental::builtin_mpi_type M, typename C>
        requires std::is_base_of_v<C, T>
    struct_builder& field(M C::* member) & {
        add(member_of_sample(member), mpi::experimental::builtin_datatype<M>(), 1, sizeof(M));
        return *this;
    }

    /// @copydoc field(M C::*) &
    template <mpi::experimental::builtin_mpi_type M, typename C>
        requires std::is_base_of_v<C, T>
    struct_builder&& field(M C::* member) && {
        return std::move(field(member));
    }

    /// @brief Adds a member of any type as one element of `dt`, which describes the whole member.
    template <typename M, typename C, typename Datatype>
        requires std::is_base_of_v<C, T> && (!std::is_function_v<M>)
                 && mpi::experimental::convertible_to_mpi_handle<std::remove_cvref_t<Datatype>, MPI_Datatype>
    struct_builder& field(M C::* member, Datatype&& dt) & {
        add(member_of_sample(member), keep(std::forward<Datatype>(dt)), 1, sizeof(M));
        return *this;
    }

    /// @copydoc field(M C::*, Datatype&&) &
    template <typename M, typename C, typename Datatype>
        requires std::is_base_of_v<C, T> && (!std::is_function_v<M>)
                 && mpi::experimental::convertible_to_mpi_handle<std::remove_cvref_t<Datatype>, MPI_Datatype>
    struct_builder&& field(M C::* member, Datatype&& dt) && {
        return std::move(field(member, std::forward<Datatype>(dt)));
    }

    /// @brief Adds an array member (a C array or `std::array`) of a builtin type as a block of all its elements.
    template <detail::array_member M, typename C>
        requires std::is_base_of_v<C, T>
                 && mpi::experimental::builtin_mpi_type<
                     typename detail::array_member_traits<std::remove_cv_t<M>>::element_type>
    struct_builder& array_field(M C::* member) & {
        using traits = detail::array_member_traits<std::remove_cv_t<M>>;
        add(member_of_sample(member),
            mpi::experimental::builtin_datatype<typename traits::element_type>(),
            array_length<traits>(),
            sizeof(M));
        return *this;
    }

    /// @copydoc array_field(M C::*) &
    template <detail::array_member M, typename C>
        requires std::is_base_of_v<C, T>
                 && mpi::experimental::builtin_mpi_type<
                     typename detail::array_member_traits<std::remove_cv_t<M>>::element_type>
    struct_builder&& array_field(M C::* member) && {
        return std::move(array_field(member));
    }

    /// @brief Adds an array member (a C array or `std::array`) as a block of all its elements, each one element of
    /// `element_dt`.
    template <detail::array_member M, typename C, typename Datatype>
        requires std::is_base_of_v<C, T>
                 && mpi::experimental::convertible_to_mpi_handle<std::remove_cvref_t<Datatype>, MPI_Datatype>
    struct_builder& array_field(M C::* member, Datatype&& element_dt) & {
        using traits = detail::array_member_traits<std::remove_cv_t<M>>;
        add(member_of_sample(member), keep(std::forward<Datatype>(element_dt)), array_length<traits>(), sizeof(M));
        return *this;
    }

    /// @copydoc array_field(M C::*, Datatype&&) &
    template <detail::array_member M, typename C, typename Datatype>
        requires std::is_base_of_v<C, T>
                 && mpi::experimental::convertible_to_mpi_handle<std::remove_cvref_t<Datatype>, MPI_Datatype>
    struct_builder&& array_field(M C::* member, Datatype&& element_dt) && {
        return std::move(array_field(member, std::forward<Datatype>(element_dt)));
    }

    /// @brief Adds the member `f` of the sample object as one element of `dt`, for generic code that visits the
    /// members of the sample, e.g. with Boost.PFR.
    template <typename F, typename Datatype>
        requires mpi::experimental::convertible_to_mpi_handle<std::remove_cvref_t<Datatype>, MPI_Datatype>
    struct_builder& field_of_sample(F const& f, Datatype&& dt) & {
        auto const* begin = reinterpret_cast<std::byte const*>(_sample);
        auto const* addr  = reinterpret_cast<std::byte const*>(std::addressof(f));
        KAMPING_V2_ASSERT(
            std::less_equal<>{}(begin, addr) && std::less_equal<>{}(addr + sizeof(F), begin + sizeof(T)),
            "struct_builder::field_of_sample: f is not a member of the sample object"
        );
        add(std::addressof(f), keep(std::forward<Datatype>(dt)), 1, sizeof(F));
        return *this;
    }

    /// @copydoc field_of_sample(F const&, Datatype&&) &
    template <typename F, typename Datatype>
        requires mpi::experimental::convertible_to_mpi_handle<std::remove_cvref_t<Datatype>, MPI_Datatype>
    struct_builder&& field_of_sample(F const& f, Datatype&& dt) && {
        return std::move(field_of_sample(f, std::forward<Datatype>(dt)));
    }

    /// @brief The datatype of the fields added so far, resized to `sizeof(T)` (`MPI_Type_create_struct`,
    /// `MPI_Type_create_resized`). The builder can be used further.
    /// @return The new datatype, owned and not committed.
    /// @throws mpi_error if `MPI_Type_create_struct` or `MPI_Type_create_resized` fails.
    [[nodiscard]] mpi::experimental::uncommitted_datatype build() const {
        return mpi::experimental::type_create_resized(
            mpi::experimental::type_create_struct(_lengths, _displacements, _types),
            0,
            static_cast<MPI_Aint>(sizeof(T))
        );
    }

private:
    template <typename M, typename C>
    void const* member_of_sample(M C::* member) const {
        return std::addressof(_sample->*static_cast<M T::*>(member));
    }

    template <typename Traits>
    static constexpr int array_length() {
        static_assert(Traits::length <= static_cast<std::size_t>(INT_MAX), "array member too long for MPI");
        return static_cast<int>(Traits::length);
    }

    /// Returns the handle of `dt`, taking ownership of a temporary owning datatype.
    template <typename Datatype>
    MPI_Datatype keep(Datatype&& dt) {
        using plain            = std::remove_cvref_t<Datatype>;
        MPI_Datatype const raw = mpi::experimental::handle(dt);
        if constexpr (
            !std::is_lvalue_reference_v<Datatype>
            && (std::same_as<plain, mpi::experimental::uncommitted_datatype>
                || std::same_as<plain, mpi::experimental::datatype>)
        ) {
            _owned.push_back(mpi::experimental::uncommitted_datatype::from_native(dt.release()));
        }
        return raw;
    }

    void add(void const* addr, MPI_Datatype dt, int count, [[maybe_unused]] std::size_t member_size) {
        KAMPING_V2_ASSERT(
            detail::fits_into_member(dt, count, member_size),
            "struct_builder: the datatype of a field reaches beyond its member"
        );
        MPI_Aint base   = 0;
        MPI_Aint member = 0;
        int      err    = MPI_Get_address(_sample, &base);
        if (err != MPI_SUCCESS) {
            throw mpi::experimental::mpi_error(err);
        }
        err = MPI_Get_address(addr, &member);
        if (err != MPI_SUCCESS) {
            throw mpi::experimental::mpi_error(err);
        }
        _lengths.push_back(count);
        _displacements.push_back(MPI_Aint_diff(member, base));
        _types.push_back(dt);
    }

    std::unique_ptr<T>                                   _owned_sample; // stable address when the builder is moved
    T const*                                             _sample;
    std::vector<int>                                     _lengths;
    std::vector<MPI_Aint>                                _displacements;
    std::vector<MPI_Datatype>                            _types;
    std::vector<mpi::experimental::uncommitted_datatype> _owned;
};

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
