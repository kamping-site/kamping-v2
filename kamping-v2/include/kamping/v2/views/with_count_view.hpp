// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

template <typename Base>
class with_count_view : public view_interface<with_count_view<Base>> {
    Base           base_;
    std::ptrdiff_t count_;

public:
    constexpr Base const& base() const& noexcept {
        return base_;
    }
    constexpr Base& base() & noexcept {
        return base_;
    }

    template <typename R>
    with_count_view(R&& base, std::ptrdiff_t count) : base_(kamping::v2::all(std::forward<R>(base))),
                                                      count_(count) {}

    constexpr std::ptrdiff_t mpi_count() const {
        return count_;
    }
};

template <typename R>
with_count_view(R&&, std::ptrdiff_t) -> with_count_view<kamping::v2::all_t<R>>;

template <typename Base>
inline constexpr bool enable_borrowed_buffer<with_count_view<Base>> = enable_borrowed_buffer<Base>;

} // namespace kamping::v2

namespace kamping::v2::views {

// Useful for non-range objects that expose mpi_ptr() but no size — compose with with_type
// to build a complete data_buffer. For limiting the element count of a range, prefer std::views::take.
inline constexpr kamping::v2::adaptor<1, decltype([](auto&& r, std::ptrdiff_t count) {
                                          return kamping::v2::with_count_view(std::forward<decltype(r)>(r), count);
                                      })>
    with_count{};

} // namespace kamping::v2::views
