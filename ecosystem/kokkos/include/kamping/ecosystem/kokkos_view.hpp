#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#ifdef KAMPING_HAS_KOKKOS_COMM
#include <KokkosComm/concepts.hpp>
#include <KokkosComm/impl/contiguous.hpp>
#endif
#include <Kokkos_Core.hpp>
#include <kamping/kassert/kassert.hpp>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/resize_view.hpp"

namespace kamping::v2 {

/// Wraps a Kokkos::View and presents it as an MPI data buffer.
///
/// Always stores an owned copy of the Kokkos::View handle (cheap: ref-counted pointer).
/// If the wrapped view is already contiguous this wrapper does nothing extra.
///
/// Send path: mpi_count()/mpi_ptr() const lazily pack the wrapped view into a contiguous
///            buffer on first access via deep_copy().
/// Recv path: mpi_resize_for_receive(n) resizes the wrapped rank-1 view and resets pack
///            state; compose with | resize for automatic sizing via infer().
///            mpi_ptr() returns a mutable pointer; unwrap()/operator* trigger deep_copy back.
///
/// When built with KokkosComm (KAMPING_HAS_KOKKOS_COMM defined), the concept constraint
/// and contiguous-buffer type are delegated to KokkosComm::Impl; otherwise a plain
/// Kokkos::View with the same layout and memory space is used.
template <typename T>
#ifdef KAMPING_HAS_KOKKOS_COMM
    requires KokkosComm::KokkosView<std::remove_reference_t<T>>
#else
    requires Kokkos::is_view_v<std::remove_reference_t<T>>
#endif
class kokkos_view {
    using view_type = std::remove_reference_t<T>;

    using scalar_type     = std::remove_const_t<typename view_type::value_type>;
    using execution_space = view_type::execution_space;

    // packed_view_t is the type of the created contiguous Kokkos view
#ifdef KAMPING_HAS_KOKKOS_COMM
    using packed_view_t = KokkosComm::Impl::contiguous_view_t<view_type>;
#else
    using packed_view_t = Kokkos::View<
        typename view_type::non_const_data_type,
        typename view_type::execution_space::array_layout,
        typename view_type::memory_space>;
#endif

    mutable view_type     base_;
    mutable packed_view_t packed_storage_;

    mutable bool packed_        = false;
    mutable bool needs_unpack_  = false;
    bool         is_contiguous_ = false;

    // Create a contiguous kokkos view with type packed_view_t
    static packed_view_t make_packed(view_type const& v) {
        execution_space   exec;
        std::string const label = std::string(v.label()) + "-kamping-kokkos-view";
#ifdef KAMPING_HAS_KOKKOS_COMM
        return KokkosComm::Impl::allocate_contiguous_for(exec, label, v);
#else
        return [&exec, &v, &label]<std::size_t... Is>(std::index_sequence<Is...>) {
            return packed_view_t(
                Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, label),
                static_cast<typename packed_view_t::size_type>(v.extent(Is))...
            );
        }(std::make_index_sequence<view_type::rank>{});
#endif
    }

    // Copy data from the base view to the packed_view_t
    void pack() const
        requires requires(view_type from, packed_view_t to) { Kokkos::deep_copy(to, from); }
    {
        packed_storage_ = make_packed(base_);
        Kokkos::deep_copy(packed_storage_, base_);
        packed_       = true;
        needs_unpack_ = true;
    }

    // Copy data from the packed_view_t back to the base view
    void unpack() const
        requires requires(view_type to, packed_view_t from) { Kokkos::deep_copy(to, from); }
    {
        Kokkos::deep_copy(base_, packed_storage_);
        needs_unpack_ = false;
        packed_       = false;
    }

public:
    using value_type = scalar_type;

    explicit kokkos_view(view_type const& view)
        : base_(view),
          is_contiguous_(view.span_is_contiguous()) {}

    explicit kokkos_view(view_type&& view)
        : base_(std::move(view)),
          is_contiguous_(base_.span_is_contiguous()) {}

    /// Triggers deep copy back to the base view if needed without returning a reference.
    /// Equivalent to `(void)**this` but expresses intent at call sites that only care
    /// about the side effect (e.g. immediately followed by `operator->`).
    void unwrap() const {
        if (!is_contiguous_ && needs_unpack_)
            unpack();
    }

    /// Dereference to the wrapped object, triggering deep copy back to the base view if needed.
    view_type const& operator*() const {
        unwrap();
        return base_;
    }

    /// \overload
    view_type& operator*() {
        return const_cast<view_type&>(std::as_const(*this).operator*());
    }

    /// Arrow operator; triggers deep copy back to the base view if needed.
    view_type* operator->() {
        return std::addressof(**this);
    }

    /// \overload
    view_type const* operator->() const {
        return std::addressof(**this);
    }

    // ---- Recv-side protocol -----------------------------------------------

    /// Resize the wrapped view to hold n elements and reset pack state.
    /// Only available for rank-1 views that support Kokkos::resize.
    /// Used by resize_for_receive() to implement the | resize protocol.
    void mpi_resize_for_receive(std::ptrdiff_t n)
        requires(view_type::rank == 1
                 && requires(view_type& v, typename view_type::size_type m) { Kokkos::resize(v, m); })
    {
        Kokkos::resize(base_, static_cast<typename view_type::size_type>(n));
        packed_       = false;
        needs_unpack_ = false;
    }

    // ---- MPI protocol methods --------------------------------------------

    /// Returns the number of `MPI_BYTE` elements to send or receive.
    std::ptrdiff_t mpi_count() const {
        return static_cast<std::ptrdiff_t>(base_.size());
    }

    /// Returns the MPI_Datatype corresponding to the view's scalar type.
    /// Only available for builtin types
    MPI_Datatype mpi_type() const noexcept
        requires kamping::types::is_builtin_type_v<scalar_type>
    {
        return kamping::types::builtin_type<scalar_type>().data_type();
    }

    /// Send-side pointer. Deep copy to intermediate kokkos view lazily on first call;
    /// Returns `void const*` so that a const view satisfies `send_buffer` but not `recv_buffer`.
    void const* mpi_ptr() const {
        if (is_contiguous_) {
            execution_space{}.fence();
            return base_.data();
        }
        if (!needs_unpack_ && !packed_)
            pack();
        return packed_storage_.data();
    }

    /// Recv-side pointer. Returns `void*` so MPI can write directly into it.
    /// A non-const view satisfies both `send_buffer` and `recv_buffer`.
    void* mpi_ptr() {
        if (is_contiguous_) {
            execution_space{}.fence();
            return base_.data();
        }
        if (!packed_) {
            packed_storage_ = make_packed(base_);
            packed_         = true;
            needs_unpack_   = true;
        }
        return packed_storage_.data();
    }
};

template <typename T>
kokkos_view(T const&) -> kokkos_view<T>;

template <typename T>
    requires(!std::is_lvalue_reference_v<T>)
kokkos_view(T&&) -> kokkos_view<T>;

/// Propagate supports_matched_probe from the underlying Kokkos::View type.
/// Specialize supports_matched_probe for your Kokkos::View type to control
/// whether infer() uses MPI_Mprobe or falls back to MPI_Probe + MPI_Recv.
template <typename T>
inline constexpr bool supports_matched_probe<kokkos_view<T>> =
    supports_matched_probe<std::remove_reference_t<T>>;

} // namespace kamping::v2

namespace kamping::v2::views {

inline constexpr struct kokkos_fn : kamping::v2::adaptor_closure<kokkos_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return kamping::v2::kokkos_view(std::forward<R>(r));
    }
} kokkos{};

/// Returns a resize_view wrapping an owning rank-1 kokkos_view for automatic-size receive.
/// The underlying Kokkos::View is accessible via .underlying() on the result.
template <typename T>
auto auto_kokkos_view(std::string const& label) {
    using view_t = Kokkos::View<T*, Kokkos::LayoutRight>;
    return kamping::v2::resize_view(kamping::v2::kokkos_view<view_t>(view_t(label, 0)));
}

/// Returns a resize_view wrapping an owning rank-1 kokkos_view for automatic-size receive.
/// Template parameter is the element type, e.g. auto_kokkos_view<int>().
template <typename T>
auto auto_kokkos_view() {
    return auto_kokkos_view<T>("kamping-auto-kokkos-view");
}

} // namespace kamping::v2::views
