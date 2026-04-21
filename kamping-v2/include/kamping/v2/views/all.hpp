#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

/// Non-owning view over an lvalue. T is unconstrained — supports both ranges and
/// non-range buffer types (e.g. a struct with mpi_ptr()/mpi_count() but no begin()/end()).
/// All MPI protocol methods (mpi_ptr, mpi_count, mpi_type, and future mpi_resize_for_receive)
/// are forwarded automatically via view_interface.
template <typename T>
class ref_view : public view_interface<ref_view<T>> {
    T* r_;

public:
    constexpr explicit ref_view(T& r) noexcept : r_(std::addressof(r)) {}

    constexpr T& base() const noexcept {
        return *r_;
    }
};

/// Owning view over an rvalue. Moves the object in and owns it.
/// Same protocol forwarding as ref_view via view_interface.
template <std::movable T>
class owning_view : public view_interface<owning_view<T>> {
    T r_;

public:
    constexpr explicit owning_view(T&& r) noexcept(std::is_nothrow_move_constructible_v<T>) : r_(std::move(r)) {}

    owning_view(owning_view const&)            = delete;
    owning_view& operator=(owning_view const&) = delete;
    owning_view(owning_view&&)                 = default;
    owning_view& operator=(owning_view&&)      = default;

    constexpr T& base() & noexcept {
        return r_;
    }
    constexpr T const& base() const& noexcept {
        return r_;
    }
    constexpr T&& base() && noexcept {
        return std::move(r_);
    }
};

/// Wraps a value in the appropriate view, mirroring std::views::all extended for kamping:
///   - std::ranges::view (span, string_view, …) → copy/move through; no extra wrapping to
///     avoid libc++ constraint circularity bugs with these types.
///   - kamping view (derived from view_interface_base, may be owning/non-copyable):
///       lvalue → ref_view (borrow), rvalue → move through (no wrapping needed).
///   - any other lvalue                          → ref_view
///   - any other rvalue                          → owning_view
template <typename R>
constexpr auto all(R&& r) {
    using T = std::remove_cvref_t<R>;
    if constexpr (std::derived_from<T, view_interface_base>) {
        // Kamping views (checked first — in libc++, view_interface makes them satisfy
        // std::ranges::view too, so we must intercept them before that branch).
        // Copyable kamping views (e.g. ref_view): copy/move through, same as std views.
        // Non-copyable kamping views (e.g. owning_view): lvalue → borrow via ref_view,
        //   rvalue → move through (no wrapping needed).
        if constexpr (std::is_lvalue_reference_v<R> && !std::copyable<T>)
            return ref_view<T>{r};
        else
            return std::forward<R>(r);
    } else if constexpr (std::ranges::view<T>)
        // std library views (span, string_view, …): lightweight non-owning, always copyable.
        // Copy/move through without wrapping (avoids libc++ constraint circularity bugs).
        return T(std::forward<R>(r));
    else if constexpr (std::is_lvalue_reference_v<R>)
        return ref_view<std::remove_reference_t<R>>{r};
    else
        return owning_view<T>{std::forward<R>(r)};
}

template <typename R>
using all_t = decltype(all(std::declval<R>()));

// ref_view is non-owning — always borrowed.
// owning_view owns its data — not borrowed (default false is correct).
template <typename T>
inline constexpr bool enable_borrowed_buffer<ref_view<T>> = true;

} // namespace kamping::v2
