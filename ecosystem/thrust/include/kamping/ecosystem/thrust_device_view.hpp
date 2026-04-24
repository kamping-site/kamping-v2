#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include <mpi.h>
#include <thrust/device_vector.h>

#include <kamping/types/builtin_types.hpp>

#include "kamping/v2/views/adaptor.hpp"

namespace kamping::v2 {

/// Wraps a `thrust::device_vector<T>` and presents it through the MPI buffer protocol.
///
/// The pointer returned by `mpi_ptr()` is the raw device pointer, so CUDA-aware MPI can
/// DMA directly into/out of device memory. No host staging happens in this wrapper.
///
/// V is the wrapped vector type: a (possibly const) lvalue reference for non-owning
/// wrappers, or a value type for owning wrappers — matching the ownership pattern used
/// by the kokkos_view adapter.
///
/// The `is_resizable` template flag enables `set_recv_count(n)` for the recv path.
/// The resize happens lazily inside `mpi_ptr()` so the pointer captured by MPI reflects
/// the resized buffer.
template <typename V, bool is_resizable = false>
class thrust_device_view {
    static constexpr bool is_owning = !std::is_lvalue_reference_v<V>;
    using vector_type               = std::remove_reference_t<V>;
    using value_type                = typename vector_type::value_type;
    using size_type                 = typename vector_type::size_type;
    using stored_t                  = std::conditional_t<is_owning, vector_type, vector_type*>;

    mutable stored_t _base;
    mutable bool     _needs_resize = false;
    std::ptrdiff_t   _recv_count   = 0;

    vector_type& base_ref() const noexcept {
        if constexpr (is_owning)
            return _base;
        else
            return *_base;
    }

public:
    explicit thrust_device_view(vector_type& vec)
        requires(!is_owning)
        : _base(&vec) {}

    explicit thrust_device_view(vector_type&& vec)
        requires(is_owning)
        : _base(std::move(vec)) {}

    vector_type& operator*() {
        return base_ref();
    }
    vector_type const& operator*() const {
        return base_ref();
    }
    vector_type* operator->() {
        return std::addressof(base_ref());
    }
    vector_type const* operator->() const {
        return std::addressof(base_ref());
    }

    /// Late-bound recv count for deferred recv buffers. Commits on the next `mpi_ptr()` call.
    void set_recv_count(std::ptrdiff_t n)
        requires(is_resizable)
    {
        if (n == static_cast<std::ptrdiff_t>(base_ref().size())) {
            return;
        }
        _recv_count   = n;
        _needs_resize = true;
    }

    std::ptrdiff_t mpi_count() const {
        if (_needs_resize) {
            return _recv_count;
        }
        return static_cast<std::ptrdiff_t>(base_ref().size());
    }

    MPI_Datatype mpi_type() const
        requires kamping::types::is_builtin_type_v<value_type>
    {
        return kamping::types::builtin_type<value_type>::data_type();
    }

    void* mpi_ptr() const {
        if constexpr (is_resizable) {
            if (_needs_resize) {
                base_ref().resize(static_cast<size_type>(_recv_count));
                _needs_resize = false;
            }
        }
        // thrust::device_vector::data() returns a thrust::device_ptr; .get() extracts the raw
        // device pointer. This is the pointer CUDA-aware MPI expects.
        return static_cast<void*>(base_ref().data().get());
    }
};

template <typename V>
thrust_device_view(V&) -> thrust_device_view<V&>;

template <typename V>
    requires(!std::is_lvalue_reference_v<V>)
thrust_device_view(V&&) -> thrust_device_view<V>;

} // namespace kamping::v2

namespace kamping::v2::views {

inline constexpr struct thrust_device_fn : kamping::v2::adaptor_closure<thrust_device_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return kamping::v2::thrust_device_view(std::forward<R>(r));
    }
} thrust_device{};

/// Returns an owning, resizable thrust_device_view suitable for receive operations with
/// an unknown size. The element type is the template parameter.
template <typename T>
auto auto_thrust_device_view() {
    using vector_t = thrust::device_vector<T>;
    return kamping::v2::thrust_device_view<vector_t, true>(vector_t{});
}

} // namespace kamping::v2::views
