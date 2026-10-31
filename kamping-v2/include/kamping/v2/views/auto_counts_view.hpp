// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <span>
#include <utility>
#include <vector>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

/// Deferred counts buffer for variadic collectives, wrapping an underlying data buffer.
///
/// Combines a data buffer (Base) with a counts container (Counts). The infer()
/// machinery pre-allocates via set_comm_size() (when resize=true) and writes
/// directly into counts().data() via MPI — no copy needed.
///
/// Displacements are intentionally not provided here; compose with with_displs()
/// or auto_displs() to attach them, e.g.:
///   vec | auto_counts() | auto_displs()
///
/// Template parameters:
///   Base   — the wrapped data buffer (forwarded through base())
///   Counts — the wrapped counts range (after all() wrapping)
///   resize — if true, set_comm_size() resizes via resize_for_receive(); if false,
///            the buffer must already have the right size before infer() runs.
template <typename Base, mpi::experimental::count_range Counts, bool resize = false>
    requires(!resize || has_resize<Counts> || has_mpi_resize_for_receive<Counts>)
class auto_counts_view : public view_interface<auto_counts_view<Base, Counts, resize>> {
    Base   _base;
    Counts _counts;

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

    /// Construct from a data buffer and counts buffer (no resize; counts must be pre-sized).
    template <typename R, typename C>
    auto_counts_view(R&& base, C&& counts)
        : _base(kamping::v2::all(std::forward<R>(base))),
          _counts(kamping::v2::all(std::forward<C>(counts))) {}

    /// Construct from a data buffer and counts buffer with resize enabled (tag dispatch).
    template <typename R, typename C>
    auto_counts_view(kamping::v2::resize_t, R&& base, C&& counts)
        : _base(kamping::v2::all(std::forward<R>(base))),
          _counts(kamping::v2::all(std::forward<C>(counts))) {}

    /// Pre-allocates or resizes the counts buffer for comm_size processes.
    /// Only resizes when resize=true; otherwise this is a no-op (buffer must already
    /// have the correct size).
    void set_comm_size(int n) {
        if constexpr (resize) {
            kamping::v2::resize_for_receive(_counts, static_cast<std::ptrdiff_t>(n));
        }
    }

    /// Signals that MPI has finished writing into mpi_counts(). Nothing to do here; views stacked on top (e.g.
    /// `auto_displs_view`) use it to invalidate state derived from the counts.
    void commit_counts() {}

    /// Mutable access — used by infer() to write per-rank counts directly.
    std::span<int> mpi_counts() {
        return {_counts};
    }

    /// Read-only access — used by MPI wrappers after counts are committed.
    std::span<int const> mpi_counts() const {
        return {_counts};
    }
};

template <typename R, typename C>
auto_counts_view(R&&, C&&) -> auto_counts_view<kamping::v2::all_t<R>, kamping::v2::all_t<C>>;

template <typename R, typename C>
auto_counts_view(kamping::v2::resize_t, R&&, C&&)
    -> auto_counts_view<kamping::v2::all_t<R>, kamping::v2::all_t<C>, true>;

template <typename Base, typename Counts, bool resize>
inline constexpr bool enable_borrowed_buffer<auto_counts_view<Base, Counts, resize>> =
    enable_borrowed_buffer<Base> && enable_borrowed_buffer<Counts>;

} // namespace kamping::v2

namespace kamping::v2::views {

/// Attaches deferred counts stored in an owned `Container`, which infer() resizes to the communicator size:
/// `buf | auto_counts()`.
template <typename Container = std::vector<int>>
constexpr auto auto_counts() {
    return kamping::v2::adaptor<1, decltype([](auto&& r, auto&& c) {
                                    return kamping::v2::auto_counts_view(
                                        kamping::v2::resize,
                                        std::forward<decltype(r)>(r),
                                        std::forward<decltype(c)>(c)
                                    );
                                })>{}(Container{});
}

/// Attaches deferred counts stored in a caller-provided buffer, which must already have one entry per rank:
/// `buf | auto_counts(counts)`.
template <typename C>
    requires std::ranges::range<std::remove_cvref_t<C>>
constexpr auto auto_counts(C&& counts) {
    return kamping::v2::adaptor<1, decltype([](auto&& r, auto&& c) {
                                    return kamping::v2::auto_counts_view(
                                        std::forward<decltype(r)>(r),
                                        std::forward<decltype(c)>(c)
                                    );
                                })>{}(std::forward<C>(counts));
}

/// Attaches deferred counts stored in a caller-provided buffer, which infer() resizes to the communicator size:
/// `buf | auto_counts(resize, counts)`.
template <typename C>
    requires std::ranges::range<std::remove_cvref_t<C>>
constexpr auto auto_counts(kamping::v2::resize_t, C&& counts) {
    return kamping::v2::adaptor<1, decltype([](auto&& r, auto&& c) {
                                    return kamping::v2::auto_counts_view(
                                        kamping::v2::resize,
                                        std::forward<decltype(r)>(r),
                                        std::forward<decltype(c)>(c)
                                    );
                                })>{}(std::forward<C>(counts));
}

} // namespace kamping::v2::views
