#pragma once

#include <thrust/device_vector.h>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

/// View adaptor that redirects mpi_ptr() to the raw device pointer of the wrapped
/// Thrust buffer. All other MPI buffer protocol methods (count, type, resize) pass
/// through unchanged via view_interface, so this view composes freely with
/// views::resize, views::with_type, etc.
///
/// Intended use: pipe a thrust::device_vector through this adaptor so CUDA-aware
/// MPI can DMA directly into/out of device memory without host staging.
///
/// @code
///   // send from device
///   kamping::v2::send(d_vec | kamping::v2::views::thrust, 1, comm);
///
///   // receive into pre-sized device buffer
///   kamping::v2::recv(d_recv | kamping::v2::views::thrust, 0, comm);
///
///   // receive with automatic resize (size probed via MPI_Probe)
///   kamping::v2::recv(d_recv | kamping::v2::views::thrust | kamping::v2::views::resize, 0, comm);
/// @endcode
template <typename Base>
class device_ptr_view : public view_interface<device_ptr_view<Base>> {
    Base base_;

public:
    template <typename R>
    explicit device_ptr_view(R&& base) : base_(kamping::v2::all(std::forward<R>(base))) {}

    constexpr Base& base() & noexcept {
        return base_;
    }
    constexpr Base const& base() const& noexcept {
        return base_;
    }

    /// Returns the raw device pointer via thrust::device_ptr::get().
    /// Overrides view_interface::mpi_ptr(); count, type, and resize propagate
    /// from base() as usual.
    auto mpi_ptr() {
        return this->underlying().data().get();
    }

    auto mpi_ptr() const {
        return this->underlying().data().get();
    }
};

template <typename R>
device_ptr_view(R&&) -> device_ptr_view<kamping::v2::all_t<R>>;

template <typename Base>
inline constexpr bool enable_borrowed_buffer<device_ptr_view<Base>> = enable_borrowed_buffer<Base>;

// supports_matched_probe propagates from Base via view_interface. If your MPI implementation
// does not support GPU-aware matched receives (e.g. Intel MPI with MPI_Mrecv on device
// memory), opt out by specializing for your buffer type before including this header:
//
//   template <typename T, typename Alloc>
//   inline constexpr bool kamping::v2::supports_matched_probe<thrust::device_vector<T, Alloc>> = false;

} // namespace kamping::v2

namespace kamping::v2::views {

/// Pipe adaptor: redirects mpi_ptr() to the raw Thrust device pointer.
///
///   d_vec | kamping::v2::views::thrust
inline constexpr struct thrust_fn : kamping::v2::adaptor_closure<thrust_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return kamping::v2::device_ptr_view(std::forward<R>(r));
    }
} thrust{};

} // namespace kamping::v2::views
