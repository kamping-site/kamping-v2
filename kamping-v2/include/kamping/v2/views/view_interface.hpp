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
} // namespace detail

template <typename Derived>
struct view_interface : public view_interface_base, public std::ranges::view_interface<Derived> {
    constexpr Derived& derived() noexcept {
        return static_cast<Derived&>(*this);
    }

    constexpr Derived const& derived() const noexcept {
        return static_cast<Derived const&>(*this);
    }

    constexpr auto begin()
        requires detail::has_base_range<Derived>
    {
        return std::ranges::begin(derived().base());
    }

    constexpr auto end()
        requires detail::has_base_range<Derived>
    {
        return std::ranges::end(derived().base());
    }

    constexpr auto begin() const
        requires detail::has_const_base_range<Derived>
    {
        return std::ranges::begin(derived().base());
    }

    constexpr auto end() const
        requires detail::has_const_base_range<Derived>
    {
        return std::ranges::end(derived().base());
    }

    template <typename _Derived = Derived>
    auto mpi_type() const
        requires mpi::experimental::has_mpi_type<decltype(derived().base())>
    {
        return mpi::experimental::type(derived().base());
    }

    constexpr auto mpi_count() const
        requires mpi::experimental::has_mpi_count<decltype(derived().base())>
    {
        return mpi::experimental::count(derived().base());
    }

    constexpr auto mpi_ptr()
        requires mpi::experimental::has_mpi_ptr<decltype(derived().base())>
    {
        return mpi::experimental::ptr(derived().base());
    }

    constexpr auto mpi_ptr() const
        requires mpi::experimental::has_mpi_ptr<decltype(derived().base())>
    {
        return mpi::experimental::ptr(derived().base());
    }

    constexpr auto mpi_counts()
        requires mpi::experimental::has_mpi_counts_mutable<decltype(derived().base())>
    {
        return mpi::experimental::counts(derived().base());
    }

    constexpr auto mpi_counts() const
        requires mpi::experimental::has_mpi_counts<decltype(derived().base())>
    {
        return mpi::experimental::counts(derived().base());
    }

    constexpr auto mpi_displs() const
        requires mpi::experimental::has_mpi_displs<decltype(derived().base())>
    {
        return mpi::experimental::displs(derived().base());
    }

    void mpi_resize_for_receive(std::ptrdiff_t n)
        requires(
            kamping::v2::has_mpi_resize_for_receive<decltype(derived().base())>
            || kamping::v2::has_resize<decltype(derived().base())>
        )
    {
        kamping::v2::resize_for_receive(derived().base(), n);
    }

    void commit_counts()
        requires kamping::v2::has_commit_counts<decltype(derived().base())>
    {
        derived().base().commit_counts();
    }

    void set_comm_size(int n)
        requires kamping::v2::has_set_comm_size<decltype(derived().base())>
    {
        derived().base().set_comm_size(n);
    }

    constexpr bool displs_monotonic() const
        requires kamping::v2::has_monotonic_displs<decltype(derived().base())>
    {
        return derived().base().displs_monotonic();
    }
};

} // namespace kamping::v2
