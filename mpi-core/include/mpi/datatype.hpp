// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

/// @file
/// Handles for MPI datatypes: the owning `datatype`, which is always committed, the owning `uncommitted_datatype` for
/// building blocks, and the non-owning `datatype_view`. The type constructors that create them are in
/// mpi/type_constructors.hpp.

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

/// @return A copy of `dt` (`MPI_Type_dup`), which the caller has to free.
/// @throws mpi_error if `MPI_Type_dup` fails.
inline MPI_Datatype type_dup_native(MPI_Datatype dt) {
    MPI_Datatype newtype = MPI_DATATYPE_NULL;
    int          err     = MPI_Type_dup(dt, &newtype);
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return newtype;
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

class datatype;
class uncommitted_datatype;

/// @brief Non-owning wrapper around an `MPI_Datatype`.
///
/// Implicitly constructible from a raw `MPI_Datatype` and from a `datatype`, so functions can take a
/// `datatype_view` to accept predefined, user-managed and owned datatypes alike. It never commits or frees the
/// datatype.
class datatype_view : public datatype_accessors<datatype_view> {
public:
    /// @brief Views `dt`, which must outlive the view.
    datatype_view(MPI_Datatype dt) noexcept : _type(dt) {}

    /// @brief A copy of the datatype (`MPI_Type_dup`).
    ///
    /// Returned as an `uncommitted_datatype`, since a view may refer to a datatype that is not committed. The copy of a
    /// committed datatype is committed as well, so `std::move(u).commit()` turns it into a `datatype` at no extra cost.
    /// @pre The view does not refer to `MPI_DATATYPE_NULL`.
    /// @throws mpi_error if `MPI_Type_dup` fails.
    [[nodiscard]] uncommitted_datatype dup() const; // defined after uncommitted_datatype

    /// @return The underlying `MPI_Datatype`.
    [[nodiscard]] MPI_Datatype mpi_handle() const noexcept {
        return _type;
    }

private:
    MPI_Datatype _type;
};

/// @brief Owning RAII wrapper for a derived `MPI_Datatype` that is not committed yet.
///
/// Holds a building block for other datatypes, which has to be freed but need not be committed, or a datatype that is
/// committed later with `std::move(u).commit()`. It cannot be used as the datatype of a buffer: `views::with_type`
/// rejects it at compile time. The type constructors (`type_vector`, `type_create_resized`, ...) return it and accept
/// it as a base datatype. Move-only.
///
/// @code
/// auto const column      = type_vector(n, 1, n, MPI_DOUBLE);             // freed at the end of the scope
/// auto const column_type = type_create_resized(column, 0, sizeof(double)).commit();
/// @endcode
class uncommitted_datatype : public datatype_accessors<uncommitted_datatype> {
public:
    /// @brief Holds no datatype (`MPI_DATATYPE_NULL`).
    uncommitted_datatype() noexcept = default;

    uncommitted_datatype(uncommitted_datatype const&)            = delete;
    uncommitted_datatype& operator=(uncommitted_datatype const&) = delete;

    /// @brief Transfers ownership; `o` holds `MPI_DATATYPE_NULL` afterwards.
    uncommitted_datatype(uncommitted_datatype&& o) noexcept : _type(std::exchange(o._type, MPI_DATATYPE_NULL)) {}

    /// @brief Frees the current datatype, then transfers ownership; `o` holds `MPI_DATATYPE_NULL` afterwards.
    uncommitted_datatype& operator=(uncommitted_datatype&& o) noexcept {
        if (this != &o) {
            free_if_valid();
            _type = std::exchange(o._type, MPI_DATATYPE_NULL);
        }
        return *this;
    }

    /// @brief Frees the datatype, unless it is `MPI_DATATYPE_NULL`.
    ~uncommitted_datatype() noexcept {
        free_if_valid();
    }

    /// @brief Takes ownership of the derived datatype `dt` without committing it.
    /// @pre `dt` is `MPI_DATATYPE_NULL` or a derived datatype, not a predefined one such as `MPI_INT`, which must not
    /// be freed.
    [[nodiscard]] static uncommitted_datatype from_native(MPI_Datatype dt) noexcept {
        KAMPING_V2_ASSERT(
            dt == MPI_DATATYPE_NULL || !detail::is_predefined_datatype(dt),
            "uncommitted_datatype::from_native takes a derived datatype; predefined datatypes must not be freed"
        );
        return uncommitted_datatype(dt);
    }

    /// @brief Commits the datatype and transfers it to a `datatype`; `*this` holds `MPI_DATATYPE_NULL` afterwards.
    ///
    /// Only callable on an rvalue (`std::move(u).commit()`), so that no uncommitted handle of the committed datatype
    /// is left behind. If the commit fails, the datatype is freed before the exception is thrown.
    /// @pre The datatype is not `MPI_DATATYPE_NULL`.
    /// @throws mpi_error if `MPI_Type_commit` fails.
    [[nodiscard]] datatype commit() &&; // defined after datatype

    /// @brief A copy of the datatype (`MPI_Type_dup`), not committed either.
    /// @pre The datatype is not `MPI_DATATYPE_NULL`.
    /// @throws mpi_error if `MPI_Type_dup` fails.
    [[nodiscard]] uncommitted_datatype dup() const {
        KAMPING_V2_ASSERT(_type != MPI_DATATYPE_NULL, "dup() on an uncommitted_datatype that holds no datatype");
        return uncommitted_datatype(detail::type_dup_native(_type));
    }

    /// @brief Frees the datatype now; afterwards `*this` holds `MPI_DATATYPE_NULL`.
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

    /// @return The underlying `MPI_Datatype`, e.g. to pass it to an MPI type constructor.
    [[nodiscard]] MPI_Datatype mpi_handle() const noexcept {
        return _type;
    }

    /// @return A pointer to the underlying `MPI_Datatype`, so that a type constructor can write the new datatype
    /// into `*this`.
    /// @pre `*this` holds `MPI_DATATYPE_NULL`. Writing into it otherwise would leak the datatype it holds.
    [[nodiscard]] MPI_Datatype* mpi_handle_ptr() noexcept {
        KAMPING_V2_ASSERT(
            _type == MPI_DATATYPE_NULL,
            "uncommitted_datatype::mpi_handle_ptr: the datatype must be empty, or the datatype it holds is leaked"
        );
        return &_type;
    }

private:
    explicit uncommitted_datatype(MPI_Datatype dt) noexcept : _type(dt) {}

    void free_if_valid() noexcept {
        if (_type != MPI_DATATYPE_NULL) {
            MPI_Type_free(&_type);
            _type = MPI_DATATYPE_NULL;
        }
    }

    MPI_Datatype _type = MPI_DATATYPE_NULL;
};

/// @brief Owning RAII wrapper for a committed derived `MPI_Datatype`.
///
/// A `datatype` is always committed (or holds `MPI_DATATYPE_NULL`), so it can be used as the datatype of a buffer.
/// It is created from an `uncommitted_datatype`, as returned by the type constructors (`type_contiguous`,
/// `type_vector`, `type_create_darray`, ...), with `std::move(u).commit()`, or with `datatype::commit` from a raw
/// derived datatype, and frees the datatype on destruction. Move-only.
///
/// @code
/// auto const block_type =  // committed, freed at the end of the scope
///     type_create_darray(size, rank, gsizes, distribs, dargs, psizes, array_order::c, MPI_DOUBLE).commit();
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

    /// @brief Takes ownership of the derived datatype `dt` and commits it.
    ///
    /// Committing an already committed datatype is allowed, so this also takes ownership of a datatype that is
    /// committed already. `dt` is owned from this call on, even if it throws: if the commit fails, `dt` is freed
    /// before the exception is thrown.
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

    /// @brief A copy of the datatype (`MPI_Type_dup`), committed like the original.
    /// @pre The datatype is not `MPI_DATATYPE_NULL`.
    /// @throws mpi_error if `MPI_Type_dup` fails.
    [[nodiscard]] datatype dup() const {
        KAMPING_V2_ASSERT(_type != MPI_DATATYPE_NULL, "dup() on a datatype that holds no datatype");
        // MPI_Type_dup copies the committed state, so the copy needs no MPI_Type_commit.
        return datatype(detail::type_dup_native(_type));
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

inline uncommitted_datatype datatype_view::dup() const {
    KAMPING_V2_ASSERT(_type != MPI_DATATYPE_NULL, "dup() on a datatype_view of MPI_DATATYPE_NULL");
    return uncommitted_datatype::from_native(detail::type_dup_native(_type));
}

inline datatype uncommitted_datatype::commit() && {
    KAMPING_V2_ASSERT(_type != MPI_DATATYPE_NULL, "commit() on an uncommitted_datatype that holds no datatype");
    return datatype::commit(release());
}

/// A datatype handle that can be the datatype of a buffer: anything convertible to an `MPI_Datatype` except an
/// `uncommitted_datatype`. Raw `MPI_Datatype`s and `datatype_view`s are not checked, since MPI cannot tell whether a
/// datatype is committed.
template <typename T>
concept buffer_datatype = convertible_to_mpi_handle<std::remove_cvref_t<T>, MPI_Datatype>
                          && !std::same_as<std::remove_cvref_t<T>, uncommitted_datatype>;

/// @brief Commits a datatype that is managed by hand (`MPI_Type_commit`).
///
/// For datatypes without an owning wrapper; owned datatypes are committed with `datatype::commit` or
/// `std::move(u).commit()`. `dt` is a raw `MPI_Datatype` lvalue or a wrapper exposing a writable handle via
/// `mpi_handle_ptr()` or `handle_traits`, since MPI may write the handle back. An `uncommitted_datatype` is rejected:
/// it would hold a committed datatype afterwards; use `std::move(u).commit()` instead. Committing an already
/// committed datatype is allowed.
/// @pre `dt` is a derived datatype, not a predefined one such as `MPI_INT`.
/// @throws mpi_error if `MPI_Type_commit` fails.
template <convertible_to_mpi_handle_ptr<MPI_Datatype> Datatype>
    requires(!std::same_as<std::remove_cvref_t<Datatype>, uncommitted_datatype>)
void type_commit(Datatype& dt) {
    MPI_Datatype* h = mpi::experimental::handle_ptr(dt);
    KAMPING_V2_ASSERT(
        *h != MPI_DATATYPE_NULL && !detail::is_predefined_datatype(*h),
        "type_commit takes a derived datatype; predefined datatypes are committed already"
    );
    int err = MPI_Type_commit(h);
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

} // namespace mpi::experimental
