#pragma once

#include <thrust/device_vector.h>

#include <mpi/buffer.hpp>

/// @file
/// Non-intrusive MPI buffer support for thrust::device_vector<T>.
///
/// Including this header makes thrust::device_vector<T> satisfy send_buffer and
/// recv_buffer directly — no view piping required:
///
/// @code
///   #include "kamping/ecosystem/thrust_buffer_traits.hpp"
///
///   thrust::device_vector<int> d(8);
///   kamping::v2::send(d, 1, 0, comm);   // raw device pointer used directly
///   kamping::v2::recv(d, 0, 0, comm);   // pre-sized device recv
/// @endcode
///
/// For deferred-size receives, pipe through views::resize:
/// @code
///   kamping::v2::recv(d | kamping::v2::views::resize, 0, 0, comm);
/// @endcode
///
/// @note If your MPI implementation does not support GPU-aware matched receives
///       (e.g. Intel MPI with MPI_Mrecv on device memory), opt out of matched
///       probing before including this header:
/// @code
///   template <typename T>
///   inline constexpr bool kamping::v2::supports_matched_probe<thrust::device_vector<T>> = false;
/// @endcode

namespace mpi::experimental {

/// Specializes only ptr() — count and type fall through to the std::ranges fallbacks
/// since thrust::device_vector models sized_range with a known value_type.
template <typename T, typename Alloc>
struct buffer_traits<thrust::device_vector<T, Alloc>> {
    static T* ptr(thrust::device_vector<T, Alloc>& v) {
        return v.data().get();
    }

    static T const* ptr(thrust::device_vector<T, Alloc> const& v) {
        return v.data().get();
    }
};

} // namespace mpi::experimental
