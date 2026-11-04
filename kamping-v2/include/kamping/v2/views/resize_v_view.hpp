// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <ranges>
#include <utility>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

/// Variadic-receive counterpart of resize_view.
///
/// Wraps a base buffer that already exposes mpi_counts() (per-process counts) and
/// mpi_displs() (per-process displacements). On mpi_ptr() the underlying data
/// buffer is resized to fit all incoming elements.
///
/// The required size is max(counts[i] + displs[i]) over all i, which is correct for
/// arbitrary displacements; for monotonic displacements (see has_monotonic_displs)
/// it is displs.back() + counts.back().
///
/// Typical composition:
///   recv_buf | auto_counts() | auto_displs()            | resize_v
///   recv_buf | auto_counts() | with_displs(user_displs) | resize_v
template <typename Base>
    requires mpi::experimental::has_mpi_counts<Base> && mpi::experimental::has_mpi_displs<Base>
class resize_v_view : public view_interface<resize_v_view<Base>> {
    Base _base;

public:
    template <typename R>
    explicit resize_v_view(R&& base) : _base(kamping::v2::all(std::forward<R>(base))) {}

    constexpr Base& base() & noexcept {
        return _base;
    }
    constexpr Base const& base() const& noexcept {
        return _base;
    }
    constexpr Base&& base() && noexcept {
        return std::move(_base);
    }

    /// Eagerly resizes the underlying data buffer to fit all incoming elements,
    /// reading the (committed) counts and displacements from the base — which in turn
    /// realizes the lazily-computed displacements of the inner auto_displs/auto_counts
    /// stack. Idempotent. Called automatically by mpi_ptr(), or explicitly via
    /// kamping::v2::materialize() to realize the buffer before the MPI call.
    void materialize() {
        auto const&    counts     = mpi::experimental::counts(_base);
        auto const&    displs     = mpi::experimental::displs(_base);
        auto const*    counts_ptr = std::ranges::data(counts);
        auto const*    displs_ptr = std::ranges::data(displs);
        auto const     n          = std::ranges::size(counts);
        std::ptrdiff_t total      = 0;
        // Fast path: monotonically increasing displs (e.g. exclusive_scan or user-declared).
        // Tight O(1) bound: last_displ + last_count.
        // General path: non-monotonic displs require max(displs[i] + counts[i]) over all i.
        //
        // The compile-time check (has_monotonic_displs) selects whether the fast path is
        // available at all. The runtime check (displs_monotonic()) allows user types to
        // conditionally report non-monotonic displacements, e.g. when the same view type
        // is used with both monotonic and non-monotonic data depending on construction.
        if constexpr (has_monotonic_displs<Base>) {
            if (n > 0 && _base.displs_monotonic()) {
                total = static_cast<std::ptrdiff_t>(displs_ptr[n - 1]) + counts_ptr[n - 1];
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                total = std::max(total, static_cast<std::ptrdiff_t>(displs_ptr[i] + counts_ptr[i]));
            }
        }
        kamping::v2::resize_for_receive(_base, total);
        kamping::v2::materialize(_base);
    }

    /// Triggers the lazy resize, then returns the data pointer.
    auto mpi_ptr() {
        materialize();
        return mpi::experimental::ptr(_base);
    }
};

template <typename R>
resize_v_view(R&&) -> resize_v_view<kamping::v2::all_t<R>>;

template <typename Base>
inline constexpr bool enable_borrowed_buffer<resize_v_view<Base>> = enable_borrowed_buffer<Base>;

} // namespace kamping::v2

namespace kamping::v2::views {

/// Wraps a base buffer (which must already provide mpi_counts() and mpi_displs(),
/// e.g. via auto_counts | auto_displs or with_counts | with_displs) so the
/// underlying data buffer is resized to the correct total size on mpi_ptr().
/// Use as: buf | auto_counts() | auto_displs() | resize_v
inline constexpr struct resize_v_fn : kamping::v2::adaptor_closure<resize_v_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return kamping::v2::resize_v_view(std::forward<R>(r));
    }
} resize_v{};

} // namespace kamping::v2::views
