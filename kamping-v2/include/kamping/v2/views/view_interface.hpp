// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>

#include <mpi.h>

#include "kamping/types/builtin_types.hpp"
#include "kamping/v2/views/concepts.hpp"

namespace kamping::v2 {

// ──────────────────────────────────────────────────────────────────────────────
// View adaptor tags
// ──────────────────────────────────────────────────────────────────────────────

/// Tag used to request in-place resize of a counts or data buffer.
struct resize_t {};
inline constexpr resize_t resize{};

/// Tag for user-provided displacements that are known to be monotonically
/// non-decreasing (e.g. computed via exclusive_scan). Enables the O(1)
/// tight-bound formula displs.back() + counts.back() in resize_v_view
/// instead of the O(p) max(counts[i]+displs[i]) formula.
struct monotonic_t {};
inline constexpr monotonic_t monotonic{};

// ──────────────────────────────────────────────────────────────────────────────
// Resize helpers (from ranges/ranges.hpp)
// ──────────────────────────────────────────────────────────────────────────────

// Tag base for all kamping views. Used to guard the std::ranges::size fallback
// against ADL circularity (std::ranges::size → ADL mpi::experimental::size → std::ranges::sized_range).
struct view_interface_base {};

/// A type that is a "view" in the kamping sense: either a std::ranges::view
/// (lightweight, copyable, e.g. std::span) or a kamping view (derived from
/// view_interface_base — may be owning and non-copyable like owning_view).
/// Mirrors the role of the std::ranges::view concept but extended for kamping.
template <typename T>
concept view = std::ranges::view<T> || std::derived_from<T, view_interface_base>;

/// Type implements the custom MPI resize protocol (preferred over plain resize()).
template <typename T>
concept has_mpi_resize_for_receive = requires(T& t, std::ptrdiff_t n) { t.mpi_resize_for_receive(n); };

/// Type is a standard resizable container (e.g. std::vector).
template <typename T>
concept has_resize = requires(T& t, std::size_t n) { t.resize(n); };

/// Resize t to hold n MPI elements before a receive. Dispatches to:
///   1. t.mpi_resize_for_receive(n) — custom protocol (e.g. resize_and_overwrite, NUMA alloc)
///   2. t.resize(n)                 — standard containers
template <has_mpi_resize_for_receive T>
void resize_for_receive(T& t, std::ptrdiff_t n) {
    t.mpi_resize_for_receive(n);
}

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
    constexpr auto mpi_counts()
        requires mpi::experimental::has_mpi_counts_mutable<decltype(std::declval<Self&>().base())>
    {
        return mpi::experimental::counts(derived().base());
    }

    template <typename Self = Derived>
    constexpr auto mpi_counts() const
        requires mpi::experimental::has_mpi_counts<decltype(std::declval<Self const&>().base())>
    {
        return mpi::experimental::counts(derived().base());
    }

    template <typename Self = Derived>
    constexpr auto mpi_displs() const
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

    /// Propagates eager realization to the wrapped base, so a deferred resizing view
    /// (resize_view / resize_v_view) stays materializable when it is nested inside other
    /// view layers (with_type, GPU/serialization adaptors, …). Without this, an outer
    /// wrapper would forward mpi_ptr() — and thus the lazy resize — but silently swallow
    /// an explicit kamping::v2::materialize(), diverging the two paths.
    ///
    /// Guarded on the base having a materialize() member and calling it directly, exactly
    /// like the other forwarders (has_commit_counts / commit_counts, …). has_materialize
    /// is itself forwarded this way, so it chains through any number of wrapping layers.
    /// The deferred-buffer safety check lives solely in the free function kamping::v2::
    /// materialize(), the public entry point that gates this whole cascade.
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
    /// nested inside other view layers stays a deferred_recv_buf. Mirrors the variadic
    /// protocol forwarding above (set_comm_size / commit_counts); without it, a wrapper
    /// would forward mpi_ptr()/mpi_count() but not the inferred-size hook, so infer()
    /// could not resize the buffer and it would receive into a stale size.
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
