#pragma once

#include <sycl/sycl.hpp>
#include <type_traits>
#include <variant>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

namespace detail {
/// Extract the raw device pointer from a SYCL accessor via the interop handle.
/// Based on https://github.com/codeplaysoftware/SYCL-samples/blob/main/src/MPI_with_SYCL/send_recv_buff.cpp
template <typename Accessor>
inline void* get_device_pointer(Accessor const& acc, sycl::interop_handle const& ih) {
    void* device_ptr = nullptr;
    switch (ih.get_backend()) {
#if SYCL_EXT_ONEAPI_BACKEND_CUDA
        case sycl::backend::ext_oneapi_cuda:
            device_ptr = reinterpret_cast<void*>(ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(acc));
            break;
#endif
#if SYCL_EXT_ONEAPI_BACKEND_HIP
        case sycl::backend::ext_oneapi_hip:
            device_ptr = reinterpret_cast<void*>(ih.get_native_mem<sycl::backend::ext_oneapi_hip>(acc));
            break;
#endif
        case sycl::backend::ext_oneapi_level_zero:
            device_ptr = reinterpret_cast<void*>(ih.get_native_mem<sycl::backend::ext_oneapi_level_zero>(acc));
            break;
        default:
            throw std::runtime_error{
                "Backend does not support buffer interop required for device-aware MPI with sycl::buffer"};
    }
    return device_ptr;
}

/// Unified traits for SYCL accessor types. Provides mode and is_host for both
/// sycl::accessor (device/host_task) and sycl::host_accessor.
template <typename T>
struct sycl_accessor_traits {};

template <typename DataT, int Dims, sycl::access_mode Mode, sycl::target Tgt,
          sycl::access::placeholder PH>
struct sycl_accessor_traits<sycl::accessor<DataT, Dims, Mode, Tgt, PH>> {
    static constexpr sycl::access_mode mode     = Mode;
    static constexpr bool              is_host  = false;
};

template <typename DataT, int Dims, sycl::access_mode Mode>
struct sycl_accessor_traits<sycl::host_accessor<DataT, Dims, Mode>> {
    static constexpr sycl::access_mode mode    = Mode;
    static constexpr bool              is_host = true;
};

template <typename T>
concept writable_sycl_accessor =
    requires { sycl_accessor_traits<std::remove_cvref_t<T>>::mode; }
    && (sycl_accessor_traits<std::remove_cvref_t<T>>::mode == sycl::access_mode::write
        || sycl_accessor_traits<std::remove_cvref_t<T>>::mode == sycl::access_mode::read_write);

template <typename T>
concept readable_sycl_accessor =
    requires { sycl_accessor_traits<std::remove_cvref_t<T>>::mode; }
    && (sycl_accessor_traits<std::remove_cvref_t<T>>::mode == sycl::access_mode::read
        || sycl_accessor_traits<std::remove_cvref_t<T>>::mode == sycl::access_mode::read_write);
} // namespace detail

/// True if T is any SYCL accessor (device, host_task, or host).
template <typename T>
concept sycl_accessor = requires { detail::sycl_accessor_traits<std::remove_cvref_t<T>>::mode; };

/// True if T is a sycl::accessor (used inside kernels or host_task; requires an interop_handle
/// to access device memory).
template <typename T>
concept sycl_device_accessor =
    sycl_accessor<T> && !detail::sycl_accessor_traits<std::remove_cvref_t<T>>::is_host;

/// True if T is a sycl::host_accessor (host-side access; pointer available via get_pointer()).
template <typename T>
concept sycl_host_accessor =
    sycl_accessor<T> && detail::sycl_accessor_traits<std::remove_cvref_t<T>>::is_host;

/// Wraps a SYCL accessor and exposes it as an MPI data buffer.
///
/// Two construction modes:
///  - Device accessor (inside a host_task): construct with the accessor and sycl::interop_handle.
///    mpi_ptr() extracts the device pointer via get_native_mem() for the active backend.
///  - Host accessor: construct with the accessor only.
///    mpi_ptr() uses get_pointer(), which returns the host-side contiguous buffer pointer.
///
/// mpi_count() and mpi_type() are forwarded from the accessor via view_interface.
///
/// @tparam Base  A sycl::accessor or sycl::host_accessor specialization.
template <sycl_accessor Base>
class sycl_view : public view_interface<sycl_view<Base>> {
    // ih_ only exists for device accessors; [[no_unique_address]] eliminates it for host path.
    using ih_storage_t = std::conditional_t<sycl_device_accessor<Base>,
                                            sycl::interop_handle const*,
                                            std::monostate>;

    Base                               base_;
    [[no_unique_address]] ih_storage_t ih_;

public:
    /// Device accessor constructor. @p ih must outlive this view.
    explicit sycl_view(Base base, sycl::interop_handle const& ih)
        requires sycl_device_accessor<Base>
        : base_(std::move(base)),
          ih_(&ih) {}

    /// Host accessor constructor.
    explicit sycl_view(Base base)
        requires sycl_host_accessor<Base>
        : base_(std::move(base)),
          ih_{} {}

    /// @{
    Base& base() & noexcept {
        return base_;
    }
    Base const& base() const& noexcept {
        return base_;
    }
    /// @}

    /// Returns void const* for MPI send operations.
    void const* mpi_ptr() const
        requires detail::readable_sycl_accessor<Base>
    {
        if constexpr (sycl_host_accessor<Base>)
            return static_cast<void const*>(base_.get_pointer());
        else
            return static_cast<void const*>(detail::get_device_pointer(base_, *ih_));
    }

    /// Returns void* for MPI receive operations.
    void* mpi_ptr()
        requires detail::writable_sycl_accessor<Base>
    {
        if constexpr (sycl_host_accessor<Base>)
            return static_cast<void*>(base_.get_pointer());
        else
            return detail::get_device_pointer(base_, *ih_);
    }
};

template <typename R>
    requires sycl_device_accessor<R>
sycl_view(R&&, sycl::interop_handle const&) -> sycl_view<std::remove_cvref_t<R>>;

template <typename R>
    requires sycl_host_accessor<R>
sycl_view(R&&) -> sycl_view<std::remove_cvref_t<R>>;

/// Propagate supports_matched_probe from Base.
/// If your MPI implementation does not support GPU-aware matched receives
/// (e.g. Intel MPI with MPI_Mrecv on device memory), opt out by specializing
/// for your accessor type before including this header:
///
///   template <> inline constexpr bool
///   kamping::v2::supports_matched_probe<sycl::accessor<int, 1, sycl::access_mode::read>> = false;
template <typename Base>
inline constexpr bool supports_matched_probe<sycl_view<Base>> = supports_matched_probe<Base>;

} // namespace kamping::v2

namespace kamping::v2::views {

/// Pipe adaptor: wraps a SYCL accessor as an MPI buffer.
///
/// Device accessor (inside a host_task) — pass the interop handle:
/// @code
/// h.host_task([=](sycl::interop_handle ih) {
///     kamping::v2::send(acc | kamping::v2::views::sycl(ih), 1, 0, world);
///     kamping::v2::send(kamping::v2::views::sycl(acc, ih), 1, 0, world);  // equivalent
/// });
/// @endcode
///
/// Host accessor — use as a zero-argument closure (no handle needed):
/// @code
/// sycl::host_accessor hacc{buf, sycl::read_only};
/// kamping::v2::send(hacc | kamping::v2::views::sycl, 1, 0, world);
/// @endcode
inline constexpr struct sycl_fn : kamping::v2::adaptor_closure<sycl_fn> {
    /// Host accessor path: wraps the accessor directly.
    template <kamping::v2::sycl_host_accessor R>
    constexpr auto operator()(R&& r) const {
        return kamping::v2::sycl_view(std::forward<R>(r));
    }

    /// Device accessor path (partial): returns a pipeable closure that binds ih.
    /// Uses the standard adaptor machinery — same pattern as views::with_type(dt) etc.
    /// Not constexpr: sycl::interop_handle is a runtime-only type.
    auto operator()(sycl::interop_handle const& ih) const {
        return kamping::v2::adaptor<1, decltype([](auto&& r, sycl::interop_handle const& ih) {
                                        return kamping::v2::sycl_view(std::forward<decltype(r)>(r), ih);
                                    })>{}(ih);
    }

    /// Device accessor path (full call).
    template <kamping::v2::sycl_device_accessor R>
    constexpr auto operator()(R&& r, sycl::interop_handle const& ih) const {
        return kamping::v2::sycl_view(std::forward<R>(r), ih);
    }
} sycl{};

} // namespace kamping::v2::views
