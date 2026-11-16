// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once
#include <concepts>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/view_interface.hpp"
#include "mpi/handle.hpp"

namespace kamping::v2 {

/// Overrides the MPI datatype (`mpi_type()`) of the wrapped buffer.
///
/// `Type` is `MPI_Datatype` when the datatype is borrowed, or a move-only handle (such as `datatype`) that the view
/// owns.
template <typename Base, typename Type = MPI_Datatype>
class with_type_view : public view_interface<with_type_view<Base, Type>> {
    Base _base;
    Type _type;

public:
    constexpr Base const& base() const& noexcept {
        return _base;
    }
    constexpr Base& base() & noexcept {
        return _base;
    }
    constexpr Base&& base() && noexcept {
        return std::move(_base);
    }

    template <typename R, typename T>
    with_type_view(R&& base, T&& type) : _base(kamping::v2::all(std::forward<R>(base))),
                                         _type(std::forward<T>(type)) {}

    constexpr MPI_Datatype mpi_type() const {
        return mpi::experimental::handle(_type);
    }
};

template <typename R>
with_type_view(R&&, MPI_Datatype) -> with_type_view<kamping::v2::all_t<R>>;

// A view that owns its datatype must not be treated as borrowed: the datatype is freed with the view.
template <typename Base, typename Type>
inline constexpr bool enable_borrowed_buffer<with_type_view<Base, Type>> =
    enable_borrowed_buffer<Base> && std::same_as<Type, MPI_Datatype>;

namespace detail {
struct with_type_fn {
    // An rvalue of a move-only handle (an owning `datatype`) is moved into the view, which then owns it, like `all()`
    // does for rvalue ranges. Everything else is borrowed: the view stores the raw `MPI_Datatype`. A move-only lvalue
    // bound by a partial application arrives as a `std::reference_wrapper` (see `adaptor`).
    template <typename R, typename T>
        requires mpi::experimental::
            convertible_to_mpi_handle<std::unwrap_reference_t<std::remove_cvref_t<T>>, MPI_Datatype>
        auto operator()(R&& r, T&& type) const {
        using handle_t = std::unwrap_reference_t<std::remove_cvref_t<T>>;
        if constexpr (!std::is_lvalue_reference_v<T> && !std::is_copy_constructible_v<handle_t>) {
            return with_type_view<kamping::v2::all_t<R>, handle_t>(std::forward<R>(r), std::move(type));
        } else {
            handle_t const& h = type;
            return with_type_view<kamping::v2::all_t<R>>(std::forward<R>(r), mpi::experimental::handle(h));
        }
    }
};
} // namespace detail

} // namespace kamping::v2

namespace kamping::v2::views {

/// Overrides the MPI datatype: `buf | views::with_type(dt)`.
///
/// `dt` is an `MPI_Datatype` or a datatype handle such as `datatype` or `datatype_view`. A handle passed as an lvalue
/// is borrowed and must outlive the use of the buffer. A temporary `datatype` is moved into the view, which frees it
/// when it is destroyed:
///
/// @code
/// send(matrix | views::with_type(datatype::from_native(block)) | views::with_count(1), 0, comm);
/// @endcode
///
/// @code{.signature}
/// template <typename Type> auto with_type(Type&& type);
/// template <typename Buf, typename Type> auto with_type(Buf&& buf, Type&& type);
/// @endcode
inline constexpr kamping::v2::adaptor<1, kamping::v2::detail::with_type_fn> with_type{};

} // namespace kamping::v2::views
