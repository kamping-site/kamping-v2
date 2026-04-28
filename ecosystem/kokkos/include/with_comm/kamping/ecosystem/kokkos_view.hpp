#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/impl/contiguous.hpp>
#include <Kokkos_Core.hpp>
#include <kamping/kassert/kassert.hpp>

#include "kamping/v2/views/adaptor.hpp"


namespace kamping::v2 {

/// Wraps a Kokkos::View and packs it into a contiguous Kokkos::View.
///
/// T is the wrapped Kokkos::View type: a (possibly const) lvalue reference for non-owning
/// wrappers, or a value type for owning wrappers. If the wrapped Kokkos:View is contiguous this wrapper does nothing
///
/// Send path: mpi_size()/mpi_data() lazily create a contiguous view and deep_copy() the
///            wrapped view into it.
/// Recv path: set_recv_count(n) only works if the template parameter is_resizable is set and the wrapped
///            Kokkos::views has rank = 1
///
template <typename T, bool is_resizable = false>
    requires KokkosComm::KokkosView<std::remove_reference_t<T>>
class kokkos_view {
    static constexpr bool is_owning = !std::is_lvalue_reference_v<T>;
    using view_type                 = std::remove_reference_t<T>;

    using scalar_type     = std::remove_const_t<typename view_type::value_type>;
    using execution_space = view_type::execution_space;
    using stored_t        = std::conditional_t<is_owning, view_type, view_type*>;
    using packed_view_t   = KokkosComm::Impl::contiguous_view_t<view_type>;

    mutable stored_t      base_;
    mutable packed_view_t packed_storage_;

    mutable bool   packed_        = false;
    mutable bool   needs_unpack_  = false;
    bool           needs_resize_  = false;
    bool           is_contiguous_ = false;
    std::ptrdiff_t recv_count_    = 0;

    view_type& base_ref() const noexcept {
        if constexpr (is_owning)
            return base_;
        else
            return *base_;
    }

    static packed_view_t make_packed(view_type const& v) {
        execution_space   exec;
        std::string const label = std::string(v.label()) + "-kamping-kokkos-view";

        return KokkosComm::Impl::allocate_contiguous_for(exec, label, v);
    }

    void pack() const
        requires requires(view_type from, packed_view_t to) { Kokkos::deep_copy(to, from); }
    {
        packed_storage_ = make_packed(base_ref());
        Kokkos::deep_copy(packed_storage_, base_ref());
        packed_       = true;
        needs_unpack_ = true;
    }

    void unpack() const
        requires requires(view_type to, packed_view_t from) { Kokkos::deep_copy(to, from); }
    {
        Kokkos::deep_copy(base_ref(), packed_storage_);
        needs_unpack_ = false;
    }

public:
    explicit kokkos_view(view_type& view)
        requires(!is_owning)
        : base_(&view),
          is_contiguous_(view.span_is_contiguous()) {}

    explicit kokkos_view(view_type&& view)
        requires(is_owning)
        : base_(std::move(view)),
          is_contiguous_(base_.span_is_contiguous()) {}

    void unwrap() const {
        if (!is_contiguous_ && needs_unpack_)
            unpack();
    }

    view_type const& operator*() const {
        unwrap();
        return base_ref();
    }

    view_type& operator*() {
        return const_cast<view_type&>(std::as_const(*this).operator*());
    }

    view_type* operator->() {
        return std::addressof(**this);
    }
    view_type const* operator->() const {
        return std::addressof(**this);
    }

    void set_recv_count(std::ptrdiff_t n)
        requires(
            is_resizable && view_type::rank == 1
            && requires(view_type& v, typename view_type::size_type m) { Kokkos::resize(v, m); }
        )
    {
        packed_       = false;
        needs_unpack_ = false;

        auto const current_size = static_cast<std::ptrdiff_t>(base_ref().size());
        if (n == current_size)
            return;

        recv_count_ = n;
        KAMPING_ASSERT(current_size == 0, "Wrapped kokkos_view size must be zero for resizing");
        needs_resize_ = true;
    }

    std::ptrdiff_t mpi_count() const {
        if (needs_resize_)
            return recv_count_;
        return static_cast<std::ptrdiff_t>(base_ref().size());
    }

    MPI_Datatype mpi_type() const noexcept
        requires kamping::types::is_builtin_type_v<scalar_type>
    {
        return kamping::types::builtin_type<scalar_type>().data_type();
    }

    void const* mpi_ptr() const {
        if (is_contiguous_)
            return base_ref().data();
        if (!needs_unpack_ && !packed_)
            pack();
        return packed_storage_.data();
    }

    void* mpi_ptr() {
        if constexpr (requires(view_type& v, typename view_type::size_type m) { Kokkos::resize(v, m); }) {
            if (needs_resize_) {
                Kokkos::resize(base_ref(), static_cast<typename view_type::size_type>(recv_count_));
                needs_resize_ = false;
            }
        }
        return const_cast<void*>(std::as_const(*this).mpi_ptr());
    }
};

template <typename T>
kokkos_view(T&) -> kokkos_view<T&>;

template <typename T>
    requires(!std::is_lvalue_reference_v<T>)
kokkos_view(T&&) -> kokkos_view<T>;

} // namespace kamping::v2

namespace kamping::v2::views {
inline constexpr struct kokkos_fn : kamping::v2::adaptor_closure<kokkos_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return kamping::v2::kokkos_view(std::forward<R>(r));
    }
} kokkos{};

/// Returns an owning rank-1 kokkos_view for receive with a custom label.
template <typename T>
auto auto_kokkos_view(std::string const& label) {
    using view_t = Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace>;
    return kamping::v2::kokkos_view<view_t, true>(view_t(label, 0));
}

/// Returns an owning rank-1 kokkos_view for receive
/// Template parameter is the element type, e.g. auto_kokkos_view<int>()
template <typename T>
auto auto_kokkos_view() {
    return auto_kokkos_view<T>("kamping-auto-kokkos-view");
}

} // namespace kamping::v2::views