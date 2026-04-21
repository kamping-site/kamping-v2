#pragma once

#include <algorithm>
#include <ranges>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

/// Variadic-receive counterpart of resize_view.
///
/// Wraps a base buffer that already exposes mpi_counts() (per-process counts) and
/// mpi_displs() (per-process displacements). On mpi_data() the underlying data
/// buffer is resized to fit all incoming elements.
///
/// The required size is max(counts[i] + displs[i]) over all i — the same formula
/// as kamping::internal::compute_required_recv_buf_size_in_vectorized_communication.
/// Correct for both monotonically-increasing (auto-computed) and user-provided
/// non-monotonic displacements.
///
/// Typical composition:
///   recv_buf | with_counts(auto_counts()) | auto_displs(resize, displs) | resize_v
///   recv_buf | with_counts(auto_counts()) | with_displs(user_displs)         | resize_v
template <typename Base>
    requires mpi::experimental::has_mpi_counts<Base> && mpi::experimental::has_mpi_displs<Base>
class resize_v_view : public view_interface<resize_v_view<Base>> {
    Base base_;

public:
    template <typename R>
    explicit resize_v_view(R&& base) : base_(kamping::v2::all(std::forward<R>(base))) {}

    constexpr Base& base() & noexcept {
        return base_;
    }
    constexpr Base const& base() const& noexcept {
        return base_;
    }

    // mpi_counts, mpi_displs, mpi_type, mpi_count are all forwarded through view_interface.

    auto mpi_ptr() {
        auto const&    counts     = mpi::experimental::counts(base_);
        auto const&    displs     = mpi::experimental::displs(base_);
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
            if (n > 0 && base_.displs_monotonic()) {
                total = static_cast<std::ptrdiff_t>(displs_ptr[n - 1]) + counts_ptr[n - 1];
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                total = std::max(total, static_cast<std::ptrdiff_t>(displs_ptr[i] + counts_ptr[i]));
            }
        }
        kamping::v2::resize_for_receive(base_, total);
        return mpi::experimental::ptr(base_);
    }
};

template <typename R>
resize_v_view(R&&) -> resize_v_view<kamping::v2::all_t<R>>;

template <typename Base>
inline constexpr bool enable_borrowed_buffer<resize_v_view<Base>> = enable_borrowed_buffer<Base>;

} // namespace kamping::v2

namespace kamping::v2::views {

/// Wraps a base buffer (which must already provide mpi_counts() and mpi_displs()
/// via e.g. with_counts | auto_displs or with_counts | with_displs) so the
/// underlying data buffer is resized to the correct total size on mpi_ptr().
/// Use as: buf | with_counts(...) | auto_displs(...) | resize_v
inline constexpr struct resize_v_fn : kamping::v2::adaptor_closure<resize_v_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return kamping::v2::resize_v_view(std::forward<R>(r));
    }
} resize_v{};

} // namespace kamping::v2::views
