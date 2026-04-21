#pragma once

#include <cstddef>
#include <optional>

#include "kamping/kassert/kassert.hpp"
#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

/// Wraps a resizable container and defers the actual resize until mpi_ptr() is first accessed.
/// set_recv_count(n) is called by the collective after inferring n via infer(). mpi_count()
/// returns the stored count immediately; mpi_ptr() triggers the resize on first call.
/// mpi_type() is forwarded from the base via view_interface.
template <typename Base>
class resize_view : public view_interface<resize_view<Base>> {
    Base                          base_;
    std::optional<std::ptrdiff_t> recv_count_   = std::nullopt;
    bool                          needs_resize_ = false;

public:
    template <typename R>
    explicit resize_view(R&& base) : base_(kamping::v2::all(std::forward<R>(base))) {}

    constexpr Base& base() & noexcept {
        return base_;
    }
    constexpr Base const& base() const& noexcept {
        return base_;
    }

    /// Called by the collective with the inferred recv count. Does not resize yet.
    void set_recv_count(std::ptrdiff_t n) {
        recv_count_   = n;
        needs_resize_ = true;
    }

    /// Returns the recv count set by set_recv_count(). Overrides view_interface::mpi_count().
    std::ptrdiff_t mpi_count() const {
        if (recv_count_) {
            return *recv_count_;
        } else {
            return static_cast<std::ptrdiff_t>(mpi::experimental::count(base_));
        }
    }

    /// Triggers the lazy resize on first access, then returns the data pointer.
    /// Overrides view_interface::mpi_ptr().
    auto mpi_ptr() {
        if (needs_resize_) {
            KAMPING_ASSERT(recv_count_.has_value());
            kamping::v2::resize_for_receive(base_, *recv_count_);
            needs_resize_ = false;
        }
        return mpi::experimental::ptr(base_);
    }
};

template <typename R>
resize_view(R&&) -> resize_view<kamping::v2::all_t<R>>;

template <typename Base>
inline constexpr bool enable_borrowed_buffer<resize_view<Base>> = enable_borrowed_buffer<Base>;

} // namespace kamping::v2

namespace kamping::v2::views {

/// Wraps a resizable buffer so the collective can call set_recv_count(n) to infer its size.
/// Always outermost in a pipe chain: vec | with_type(t) | resize
/// Does not take extra arguments — use as: val | views::resize  or  views::resize(val)
inline constexpr struct resize_fn : kamping::v2::adaptor_closure<resize_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return kamping::v2::resize_view(std::forward<R>(r));
    }
} resize{};

} // namespace kamping::v2::views
