// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>

#include <mpi.h>

#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/payload.hpp"

namespace kamping::v2 {

// ──────────────────────────────────────────────────────────────────────────────
// View adaptor tags
// ──────────────────────────────────────────────────────────────────────────────

/// Tag type selecting the resizing variant of a view factory, e.g. `views::auto_counts(resize, counts)`.
struct resize_t {};
/// The @ref resize_t tag.
inline constexpr resize_t resize{};

/// Tag for user-provided displacements that are known to be monotonically
/// non-decreasing (e.g. computed via exclusive_scan). Enables the O(1)
/// tight-bound formula displs.back() + counts.back() in resize_v_view
/// instead of the O(p) max(counts[i]+displs[i]) formula.
struct monotonic_t {};
/// The @ref monotonic_t tag.
inline constexpr monotonic_t monotonic{};

// ──────────────────────────────────────────────────────────────────────────────
// Resize helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Tag base of all kamping views, used to recognize them (see @ref view and all()).
struct view_interface_base {};

/// A type that is a "view" in the kamping sense: either a std::ranges::view
/// (lightweight, copyable, e.g. std::span) or a kamping view (derived from
/// view_interface_base — may be owning and non-copyable like owning_view).
/// Mirrors the role of the std::ranges::view concept but extended for kamping.
template <typename T>
concept view = std::ranges::view<T> || std::derived_from<T, view_interface_base>;

/// Type implements the custom MPI resize protocol (preferred over plain resize()). Implementations must follow the
/// contract of @ref resize_for_receive: keep the values of the first min(old size, n) elements.
template <typename T>
concept has_mpi_resize_for_receive = requires(T& t, std::ptrdiff_t n) { t.mpi_resize_for_receive(n); };

/// Type is a standard resizable container (e.g. std::vector).
template <typename T>
concept has_resize = requires(T& t, std::size_t n) { t.resize(n); };

/// Resize t to hold n MPI elements before a receive. Dispatches to:
///   1. t.mpi_resize_for_receive(n) — custom protocol (e.g. resize_and_overwrite, NUMA alloc)
///   2. t.resize(n)                 — standard containers
///
/// **Contract.** The elements [0, min(old size, n)) keep their values; new elements are unspecified (they may be
/// left uninitialized, which is what custom protocols are for). Like a receive, which only writes the range it has
/// data for, a resize only touches what it has to. This allows growing a buffer that already holds received data,
/// e.g. to receive further messages behind it. std::vector::resize, std::string::resize_and_overwrite and
/// Kokkos::resize all behave this way.
template <has_mpi_resize_for_receive T>
void resize_for_receive(T& t, std::ptrdiff_t n) {
    t.mpi_resize_for_receive(n);
}

/// Resizes the standard container `t` to `n` elements before a receive; see the overload above for the contract.
template <typename T>
    requires(!has_mpi_resize_for_receive<T>) && has_resize<T>
void resize_for_receive(T& t, std::ptrdiff_t n) {
    t.resize(static_cast<std::size_t>(n));
}

// ──────────────────────────────────────────────────────────────────────────────
// view_interface CRTP base
// ──────────────────────────────────────────────────────────────────────────────

namespace detail {
template <typename D>
concept has_base_range = requires(D& d) {
    { d.base() } -> std::ranges::range;
};

template <typename D>
concept has_const_base_range = requires(D const& d) {
    { d.base() } -> std::ranges::range;
};

template <typename D>
concept has_base = requires(D& d) { d.base(); };

template <typename D>
concept has_const_base = requires(D const& d) { d.base(); };
} // namespace detail

/// CRTP base of all kamping views.
///
/// `Derived` implements `base()`, returning the wrapped buffer. view_interface forwards the range interface and every
/// member of the buffer protocol (`mpi_ptr()`, `mpi_count()`, `mpi_type()`, `mpi_counts()`, ...) and of the deferred
/// buffer protocols (`set_recv_count()`, `set_comm_size()`, `commit_counts()`, `materialize()`) to it, so a view only
/// implements the members it changes. A forwarder only exists if the wrapped buffer provides the member.
template <typename Derived>
struct view_interface : public view_interface_base, public std::ranges::view_interface<Derived> {
    constexpr Derived& derived() noexcept {
        return static_cast<Derived&>(*this);
    }

    constexpr Derived const& derived() const noexcept {
        return static_cast<Derived const&>(*this);
    }

    template <typename Self = Derived>
    constexpr auto begin()
        requires detail::has_base_range<Self>
    {
        return std::ranges::begin(derived().base());
    }

    template <typename Self = Derived>
    constexpr auto end()
        requires detail::has_base_range<Self>
    {
        return std::ranges::end(derived().base());
    }

    template <typename Self = Derived>
    constexpr auto begin() const
        requires detail::has_const_base_range<Self>
    {
        return std::ranges::begin(derived().base());
    }

    template <typename Self = Derived>
    constexpr auto end() const
        requires detail::has_const_base_range<Self>
    {
        return std::ranges::end(derived().base());
    }

    template <typename Self = Derived>
    auto mpi_type() const
        requires mpi::experimental::has_mpi_type<decltype(std::declval<Self const&>().base())>
    {
        return mpi::experimental::type(derived().base());
    }

    /// Forwards the value-type channel (kamping::v2::value_type(), not mpi::experimental::type())
    /// from base(), so a with_value_type annotation survives further wrapping
    /// (e.g. `... | with_value_type(dt) | resize`).
    template <typename Self = Derived>
    auto mpi_value_type() const
        requires kamping::v2::has_mpi_value_type<decltype(std::declval<Self const&>().base())>
    {
        return kamping::v2::value_type(derived().base());
    }

    template <typename Self = Derived>
    constexpr auto mpi_count() const
        requires mpi::experimental::has_mpi_count<decltype(std::declval<Self const&>().base())>
    {
        return mpi::experimental::count(derived().base());
    }

    template <typename Self = Derived>
    constexpr auto mpi_ptr()
        requires mpi::experimental::has_mpi_ptr<decltype(std::declval<Self&>().base())>
    {
        return mpi::experimental::ptr(derived().base());
    }

    template <typename Self = Derived>
    constexpr auto mpi_ptr() const
        requires mpi::experimental::has_mpi_ptr<decltype(std::declval<Self const&>().base())>
    {
        return mpi::experimental::ptr(derived().base());
    }

    template <typename Self = Derived>
    constexpr decltype(auto) mpi_counts()
        requires mpi::experimental::has_mpi_counts_mutable<decltype(std::declval<Self&>().base())>
    {
        return mpi::experimental::counts(derived().base());
    }

    template <typename Self = Derived>
    constexpr decltype(auto) mpi_counts() const
        requires mpi::experimental::has_mpi_counts<decltype(std::declval<Self const&>().base())>
    {
        return mpi::experimental::counts(derived().base());
    }

    template <typename Self = Derived>
    constexpr decltype(auto) mpi_displs() const
        requires mpi::experimental::has_mpi_displs<decltype(std::declval<Self const&>().base())>
    {
        return mpi::experimental::displs(derived().base());
    }

    template <typename Self = Derived>
    void mpi_resize_for_receive(std::ptrdiff_t n)
        requires(
            kamping::v2::has_mpi_resize_for_receive<decltype(std::declval<Self&>().base())>
            || kamping::v2::has_resize<decltype(std::declval<Self&>().base())>
        )
    {
        kamping::v2::resize_for_receive(derived().base(), n);
    }

    template <typename Self = Derived>
    void commit_counts()
        requires kamping::v2::has_commit_counts<decltype(std::declval<Self&>().base())>
    {
        derived().base().commit_counts();
    }

    /// Forwards eager realization to the wrapped base, so a deferred resizing view (resize_view /
    /// resize_v_view) can still be materialized when nested inside other view layers. Use the
    /// free function kamping::v2::materialize() rather than calling this directly.
    template <typename Self = Derived>
    void materialize()
        requires kamping::v2::has_materialize<decltype(std::declval<Self&>().base())>
    {
        derived().base().materialize();
    }

    template <typename Self = Derived>
    void set_comm_size(int n)
        requires kamping::v2::has_set_comm_size<decltype(std::declval<Self&>().base())>
    {
        derived().base().set_comm_size(n);
    }

    /// Forwards the single-count deferred protocol (set_recv_count) so a resize_view
    /// nested inside other view layers stays a deferred_recv_buf.
    template <typename Self = Derived>
    void set_recv_count(std::ptrdiff_t n)
        requires kamping::v2::deferred_recv_buf<std::remove_cvref_t<decltype(std::declval<Self&>().base())>>
    {
        derived().base().set_recv_count(n);
    }

    template <typename Self = Derived>
    constexpr bool displs_monotonic() const
        requires kamping::v2::has_monotonic_displs<decltype(std::declval<Self const&>().base())>
    {
        return derived().base().displs_monotonic();
    }

    /// Returns a reference to the innermost base that does not derive from
    /// view_interface_base, peeling through all kamping view layers.
    constexpr auto& underlying() &
        requires detail::has_base<Derived>
    {
        if constexpr (std::derived_from<std::remove_cvref_t<decltype(derived().base())>, view_interface_base>) {
            return derived().base().underlying();
        } else {
            return derived().base();
        }
    }

    /// Const overload of underlying().
    constexpr auto const& underlying() const&
        requires detail::has_const_base<Derived>
    {
        if constexpr (std::derived_from<std::remove_cvref_t<decltype(derived().base())>, view_interface_base>) {
            return derived().base().underlying();
        } else {
            return derived().base();
        }
    }

    /// Rvalue overload of underlying(): moves out of the innermost base, enabling
    /// zero-copy extraction from temporaries (e.g. result.recv.underlying()).
    constexpr auto underlying() &&
        requires detail::has_base<Derived>
    {
        if constexpr (std::derived_from<std::remove_cvref_t<decltype(derived().base())>, view_interface_base>) {
            return std::move(derived().base()).underlying();
        } else {
            return std::move(derived().base());
        }
    }
};

// Propagate supports_matched_probe through kamping view layers one step at a time via base().
// Using base() rather than underlying() ensures intermediate views (e.g. device_ptr_view
// with supports_matched_probe=false) are not skipped. Explicit per-type specializations
// (e.g. device_ptr_view) take priority over this partial specialization.
template <typename T>
    requires std::derived_from<T, view_interface_base> && detail::has_base<T>
inline constexpr bool supports_matched_probe<T> =
    supports_matched_probe<std::remove_cvref_t<decltype(std::declval<T&>().base())>>;

} // namespace kamping::v2
