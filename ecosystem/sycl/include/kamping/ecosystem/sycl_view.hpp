#pragma once

#include <sycl/sycl.hpp>
#include <type_traits>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {
  namespace detail {
    /// Get the native device pointer from a SYCL accessor
template <typename Accessor>
inline void *getDevicePointer(const Accessor &acc,
                              const sycl::interop_handle &ih) {
  void *device_ptr{nullptr};
  switch (ih.get_backend()) {
#if SYCL_EXT_ONEAPI_BACKEND_CUDA
    case sycl::backend::ext_oneapi_cuda: {
      device_ptr = reinterpret_cast<void *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(acc));
      break;
    }
#endif
#if SYCL_EXT_ONEAPI_BACKEND_HIP
    case sycl::backend::ext_oneapi_hip: {
      device_ptr = reinterpret_cast<void *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_hip>(acc));
      break;
    }
#endif
    case sycl::backend::ext_oneapi_level_zero: {
      device_ptr = reinterpret_cast<void *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_level_zero>(acc));
      break;
    }
    default: {
      throw std::runtime_error{
          "Backend does not yet support buffer interop "
          "required for device-aware MPI with sycl::buffer"};
      break;
    }
  }
  return device_ptr;
}
  }

  template <typename T>
  struct is_sycl_accessor : std::false_type {};

  template <typename DataT, int Dims, sycl::access_mode Mode,
          sycl::target Tgt, sycl::access::placeholder IsPlaceholder>
struct is_sycl_accessor<sycl::accessor<DataT, Dims, Mode, Tgt, IsPlaceholder>>
    : std::true_type {
    static constexpr auto mode = Mode;
};

  template <typename T>
concept sycl_accessor = is_sycl_accessor<std::remove_cvref_t<T>>::value;

  template <typename T>
concept writable_accessor = sycl_accessor<T> &&
    (is_sycl_accessor<std::remove_cvref_t<T>>::mode == sycl::access_mode::write ||
     is_sycl_accessor<std::remove_cvref_t<T>>::mode == sycl::access_mode::read_write);

template <typename T>
concept readable_accessor = sycl_accessor<T> &&
    (is_sycl_accessor<std::remove_cvref_t<T>>::mode == sycl::access_mode::read ||
     is_sycl_accessor<std::remove_cvref_t<T>>::mode == sycl::access_mode::read_write);

template <sycl_accessor Base>
class sycl_view : public view_interface<sycl_view<Base>> {
    Base base_;
  sycl::interop_handle const* ih_;

public:
    template <typename R>
    explicit sycl_view(R&& base, sycl::interop_handle const& ih) : base_(base), ih_(&ih) {}

    constexpr Base& base() & noexcept {
        return base_;
    }
    constexpr Base const& base() const& noexcept {
        return base_;
    }

  auto mpi_ptr() const requires readable_accessor<Base> {
    auto ptr = detail::getDevicePointer(base_, *ih_);
    return static_cast<std::add_pointer_t<std::add_const_t<std::remove_pointer_t<decltype(ptr)>>>>(ptr);
  }

  auto mpi_ptr() requires writable_accessor<Base> {
        return detail::getDevicePointer(base_, *ih_);
      }

};

template <typename R>
sycl_view(R&&, sycl::interop_handle const& ih) -> sycl_view<std::remove_cvref_t<R>>;

// template <typename Base>
// inline constexpr bool enable_borrowed_buffer<device_ptr_view<Base>> = enable_borrowed_buffer<Base>;

// supports_matched_probe propagates from Base via view_interface. If your MPI implementation
// does not support GPU-aware matched receives (e.g. Intel MPI with MPI_Mrecv on device
// memory), opt out by specializing for your buffer type before including this header:
//
//   template <typename T, typename Alloc>
//   inline constexpr bool kamping::v2::supports_matched_probe<thrust::device_vector<T, Alloc>> = false;

} // namespace kamping::v2

namespace kamping::v2::views::sycl {

/// Pipe adaptor: redirects mpi_ptr() to the raw Thrust device pointer.
///
///   d_vec | kamping::v2::views::thrust::device_ptr
// inline constexpr struct device_ptr_fn : kamping::v2::adaptor_closure<device_ptr_fn> {
    // template <typename R>
    // constexpr auto operator()(R&& r) const {
        // return kamping::v2::device_ptr_view(std::forward<R>(r));
    // }
// } device_ptr{};

} // namespace kamping::v2::views::thrust
