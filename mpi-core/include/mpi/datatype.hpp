// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

/// @file
/// Handles for MPI datatypes: the owning `datatype`, which commits a derived datatype and frees it, and the
/// non-owning `datatype_view`.

namespace mpi::experimental {

namespace detail {
/// @return Whether `dt` is a predefined datatype such as `MPI_INT`, which must not be committed by the user or freed.
/// `false` if `dt` cannot be inspected. Only used in assertions.
inline bool is_predefined_datatype(MPI_Datatype dt) noexcept {
    int num_integers  = 0;
    int num_addresses = 0;
    int num_datatypes = 0;
    int combiner      = 0;
    int err           = MPI_Type_get_envelope(dt, &num_integers, &num_addresses, &num_datatypes, &combiner);
    if (err != MPI_SUCCESS) {
        return false;
    }
    return combiner == MPI_COMBINER_NAMED;
}
} // namespace detail

/// CRTP mixin providing the read-only accessors of a datatype wrapper: `.size()`, `.extent()`, `.lower_bound()`.
/// `Derived` must implement `MPI_Datatype mpi_handle() const`.
template <typename Derived>
class datatype_accessors {
    [[nodiscard]] MPI_Datatype underlying() const noexcept {
        return static_cast<Derived const*>(this)->mpi_handle();
    }

public:
    /// @return The number of bytes of data in one element of the datatype, without gaps (`MPI_Type_size`).
    /// @throws mpi_error if `MPI_Type_size` fails.
    [[nodiscard]] int size() const {
        int s   = 0;
        int err = MPI_Type_size(underlying(), &s);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        return s;
    }

    /// @return The extent of the datatype in bytes, i.e. the distance between two consecutive elements
    /// (`MPI_Type_get_extent`).
    /// @throws mpi_error if `MPI_Type_get_extent` fails.
    [[nodiscard]] MPI_Aint extent() const {
        MPI_Aint lb     = 0;
        MPI_Aint extent = 0;
        int      err    = MPI_Type_get_extent(underlying(), &lb, &extent);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        return extent;
    }

    /// @return The lower bound of the datatype in bytes (`MPI_Type_get_extent`).
    /// @throws mpi_error if `MPI_Type_get_extent` fails.
    [[nodiscard]] MPI_Aint lower_bound() const {
        MPI_Aint lb     = 0;
        MPI_Aint extent = 0;
        int      err    = MPI_Type_get_extent(underlying(), &lb, &extent);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        return lb;
    }
};

/// @brief Non-owning wrapper around an `MPI_Datatype`.
///
/// Implicitly constructible from a raw `MPI_Datatype` and from a `datatype`, so functions can take a
/// `datatype_view` to accept predefined, user-managed and owned datatypes alike. It never commits or frees the
/// datatype.
class datatype_view : public datatype_accessors<datatype_view> {
public:
    /// @brief Views `dt`, which must outlive the view.
    datatype_view(MPI_Datatype dt) noexcept : _type(dt) {}

    /// @return The underlying `MPI_Datatype`.
    [[nodiscard]] MPI_Datatype mpi_handle() const noexcept {
        return _type;
    }

private:
    MPI_Datatype _type;
};

/// @brief Owning RAII wrapper for a derived `MPI_Datatype`.
///
/// Owns a datatype built with an MPI type constructor (`MPI_Type_contiguous`, `MPI_Type_vector`,
/// `MPI_Type_create_darray`, ...) and frees it on destruction. Move-only. `commit` takes ownership of a datatype and
/// commits it; `from_native` takes ownership without committing, e.g. of a building block for another datatype.
///
/// @code
/// MPI_Datatype block;
/// MPI_Type_create_darray(size, rank, 2, gsizes, distribs, dargs, psizes, MPI_ORDER_C, MPI_DOUBLE, &block);
/// auto const block_type = datatype::commit(block);  // committed, freed at the end of the scope
/// send(matrix | views::with_type(block_type) | views::with_count(1), 0, comm);
/// @endcode
class datatype : public datatype_accessors<datatype> {
public:
    /// @brief Holds no datatype (`MPI_DATATYPE_NULL`).
    datatype() noexcept = default;

    datatype(datatype const&)            = delete;
    datatype& operator=(datatype const&) = delete;

    /// @brief Transfers ownership; `o` holds `MPI_DATATYPE_NULL` afterwards.
    datatype(datatype&& o) noexcept : _type(std::exchange(o._type, MPI_DATATYPE_NULL)) {}

    /// @brief Frees the current datatype, then transfers ownership; `o` holds `MPI_DATATYPE_NULL` afterwards.
    datatype& operator=(datatype&& o) noexcept {
        if (this != &o) {
            free_if_valid();
            _type = std::exchange(o._type, MPI_DATATYPE_NULL);
        }
        return *this;
    }

    /// @brief Frees the datatype, unless it is `MPI_DATATYPE_NULL`.
    ~datatype() noexcept {
        free_if_valid();
    }

    /// @brief Takes ownership of the derived datatype `dt` as it is, without committing it.
    ///
    /// Use this for datatypes that are already committed, and for building blocks that are only passed to other type
    /// constructors and must be freed but need not be committed. Use @ref commit for a datatype that is used for
    /// communication.
    /// @pre `dt` is `MPI_DATATYPE_NULL` or a derived datatype, not a predefined one such as `MPI_INT`, which must not
    /// be freed.
    [[nodiscard]] static datatype from_native(MPI_Datatype dt) noexcept {
        KAMPING_V2_ASSERT(
            dt == MPI_DATATYPE_NULL || !detail::is_predefined_datatype(dt),
            "datatype::from_native takes a derived datatype; predefined datatypes must not be freed"
        );
        return datatype(dt);
    }

    /// @brief Takes ownership of the derived datatype `dt` and commits it, so that it can be used for communication.
    ///
    /// Committing an already committed datatype is allowed. `dt` is owned from this call on, even if it throws: if
    /// the commit fails, `dt` is freed before the exception is thrown.
    /// @pre `dt` is a derived datatype, not a predefined one such as `MPI_INT`, which must not be freed.
    /// @throws mpi_error if `MPI_Type_commit` fails.
    [[nodiscard]] static datatype commit(MPI_Datatype dt) {
        KAMPING_V2_ASSERT(
            dt != MPI_DATATYPE_NULL && !detail::is_predefined_datatype(dt),
            "datatype::commit takes a derived datatype; predefined datatypes must not be freed"
        );
        datatype owned(dt);
        int      err = MPI_Type_commit(&owned._type);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        return owned;
    }

    /// @brief Frees the datatype now; afterwards `*this` holds `MPI_DATATYPE_NULL`.
    ///
    /// Use this instead of the destructor to handle errors of `MPI_Type_free`. The handle is reset before the MPI
    /// call, so the destructor never frees it a second time.
    /// @throws mpi_error if `MPI_Type_free` fails.
    void free() {
        if (_type != MPI_DATATYPE_NULL) {
            MPI_Datatype tmp = std::exchange(_type, MPI_DATATYPE_NULL);
            int          err = MPI_Type_free(&tmp);
            if (err != MPI_SUCCESS) {
                throw mpi_error(err);
            }
        }
    }

    /// @brief Gives up ownership; `*this` holds `MPI_DATATYPE_NULL` afterwards.
    /// @return The datatype, which the caller has to free with `MPI_Type_free`.
    [[nodiscard]] MPI_Datatype release() noexcept {
        return std::exchange(_type, MPI_DATATYPE_NULL);
    }

    /// @brief A non-owning view of the datatype. The view does not extend the lifetime of `*this`.
    operator datatype_view() const& noexcept {
        return datatype_view{_type};
    }

    /// @brief Not available on a temporary, whose datatype would be freed while the view still refers to it.
    operator datatype_view() const&& = delete;

    /// @return The underlying `MPI_Datatype`.
    [[nodiscard]] MPI_Datatype mpi_handle() const noexcept {
        return _type;
    }

private:
    explicit datatype(MPI_Datatype dt) noexcept : _type(dt) {}

    void free_if_valid() noexcept {
        if (_type != MPI_DATATYPE_NULL) {
            MPI_Type_free(&_type);
            _type = MPI_DATATYPE_NULL;
        }
    }

    MPI_Datatype _type = MPI_DATATYPE_NULL;
};

} // namespace mpi::experimental
