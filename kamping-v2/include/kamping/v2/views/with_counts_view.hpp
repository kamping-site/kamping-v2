// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <span>
#include <utility>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

/// Attaches per-rank counts (`mpi_counts()`) to the wrapped buffer, for variadic collectives.
template <typename Base, mpi::experimental::count_range Counts>
class with_counts_view : public view_interface<with_counts_view<Base, Counts>> {
    Base   _base;
    Counts _counts;

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

    template <typename R, typename C>
    with_counts_view(R&& base, C&& counts)
        : _base(kamping::v2::all(std::forward<R>(base))),
          _counts(kamping::v2::all(std::forward<C>(counts))) {}

    constexpr Counts const& counts() const& {
        return _counts;
    }
    constexpr Counts& counts() & {
        return _counts;
    }
    constexpr Counts&& counts() && {
        return std::move(_counts);
    }

    /// Moves out the wrapped buffer and the counts.
    constexpr std::pair<Base, Counts> extract() && {
        return {std::move(_base), std::move(_counts)};
    }

    std::span<int const> mpi_counts() const {
        return {_counts};
    }
};

template <typename R, typename C>
with_counts_view(R&&, C&&) -> with_counts_view<kamping::v2::all_t<R>, kamping::v2::all_t<C>>;

template <typename Base, typename Counts>
inline constexpr bool enable_borrowed_buffer<with_counts_view<Base, Counts>> =
    enable_borrowed_buffer<Base> && enable_borrowed_buffer<Counts>;

} // namespace kamping::v2

namespace kamping::v2::views {

/// Attaches per-rank counts: `buf | views::with_counts(counts)`. The counts are borrowed if passed as lvalue.
inline constexpr kamping::v2::adaptor<1, decltype([](auto&& r, auto&& counts) {
                                          return kamping::v2::with_counts_view(
                                              std::forward<decltype(r)>(r),
                                              std::forward<decltype(counts)>(counts)
                                          );
                                      })>
    with_counts{};

} // namespace kamping::v2::views
