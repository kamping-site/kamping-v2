// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <utility>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

/// Overrides the element count (`mpi_count()`) of the wrapped buffer.
template <typename Base>
class with_count_view : public view_interface<with_count_view<Base>> {
    Base           _base;
    std::ptrdiff_t _count;

public:
    constexpr Base const& base() const& noexcept {
        return _base;
    }
    constexpr Base& base() & noexcept {
        return _base;
    }
    constexpr Base&& base() && noexcept {
        return std::move(_base);
    }

    template <typename R>
    with_count_view(R&& base, std::ptrdiff_t count) : _base(kamping::v2::all(std::forward<R>(base))),
                                                      _count(count) {}

    constexpr std::ptrdiff_t mpi_count() const {
        return _count;
    }
};

template <typename R>
with_count_view(R&&, std::ptrdiff_t) -> with_count_view<kamping::v2::all_t<R>>;

template <typename Base>
inline constexpr bool enable_borrowed_buffer<with_count_view<Base>> = enable_borrowed_buffer<Base>;

} // namespace kamping::v2

namespace kamping::v2::views {

/// Overrides the element count: `buf | views::with_count(n)`.
///
/// Useful for non-range objects that expose mpi_ptr() but no size — compose with with_type
/// to build a complete data_buffer. For limiting the element count of a range, prefer std::views::take.
inline constexpr kamping::v2::adaptor<1, decltype([](auto&& r, std::ptrdiff_t count) {
                                          return kamping::v2::with_count_view(std::forward<decltype(r)>(r), count);
                                      })>
    with_count{};

} // namespace kamping::v2::views
