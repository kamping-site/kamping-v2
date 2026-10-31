// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

#include "kamping/v2/kassert.hpp"
#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

/// Wraps a resizable container and defers the actual resize until mpi_ptr() is first accessed.
/// set_recv_count(n) is called by the collective after inferring n via infer(). mpi_count()
/// returns the stored count immediately; mpi_ptr() triggers the resize on first call.
/// mpi_type() is forwarded from the base via view_interface.
///
/// set_recv_count() may be called again after a resize, e.g. to grow the buffer behind data already received: the
/// next mpi_ptr() / materialize() resizes again, keeping the existing elements (see resize_for_receive's contract).
template <typename Base>
class resize_view : public view_interface<resize_view<Base>> {
    Base                          _base;
    std::optional<std::ptrdiff_t> _recv_count   = std::nullopt;
    bool                          _needs_resize = false;

public:
    template <typename R>
    explicit resize_view(R&& base) : _base(kamping::v2::all(std::forward<R>(base))) {}

    constexpr Base& base() & noexcept {
        return _base;
    }
    constexpr Base const& base() const& noexcept {
        return _base;
    }
    constexpr Base&& base() && noexcept {
        return std::move(_base);
    }

    /// Called by the collective with the inferred recv count. Does not resize yet.
    void set_recv_count(std::ptrdiff_t n) {
        _recv_count   = n;
        _needs_resize = true;
    }

    /// Returns the recv count set by set_recv_count(). Overrides view_interface::mpi_count().
    std::ptrdiff_t mpi_count() const {
        if (_recv_count) {
            return *_recv_count;
        } else {
            return static_cast<std::ptrdiff_t>(mpi::experimental::count(_base));
        }
    }

    /// Eagerly performs the deferred resize using the count set by set_recv_count().
    /// Idempotent: a no-op once the resize has happened. Called automatically by
    /// mpi_ptr(), or explicitly via kamping::v2::materialize() to realize the buffer
    /// before the MPI call.
    void materialize() {
        if (_needs_resize) {
            KAMPING_V2_ASSERT(_recv_count.has_value());
            kamping::v2::resize_for_receive(_base, *_recv_count);
            _needs_resize = false;
        }
        kamping::v2::materialize(_base);
    }

    /// Triggers the lazy resize on first access, then returns the data pointer.
    /// Overrides view_interface::mpi_ptr().
    auto mpi_ptr() {
        materialize();
        return mpi::experimental::ptr(_base);
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
