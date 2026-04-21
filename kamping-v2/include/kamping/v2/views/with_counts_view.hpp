#pragma once

#include <span>
#include <utility>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

template <typename Base, mpi::experimental::count_range Counts>
class with_counts_view : public view_interface<with_counts_view<Base, Counts>> {
    Base   base_;
    Counts counts_;

public:
    constexpr Base const& base() const& noexcept {
        return base_;
    }
    constexpr Base& base() & noexcept {
        return base_;
    }

    template <typename R, typename C>
    with_counts_view(R&& base, C&& counts)
        : base_(kamping::v2::all(std::forward<R>(base))),
          counts_(kamping::v2::all(std::forward<C>(counts))) {}

    constexpr Counts const& counts() const& {
        return counts_;
    }
    constexpr Counts& counts() & {
        return counts_;
    }
    constexpr Counts&& counts() && {
        return std::move(counts_);
    }

    constexpr std::pair<Base, Counts> extract() && {
        return {std::move(base_), std::move(counts_)};
    }

    std::span<int const> mpi_counts() const {
        return {counts_};
    }
};

template <typename R, typename C>
with_counts_view(R&&, C&&) -> with_counts_view<kamping::v2::all_t<R>, kamping::v2::all_t<C>>;

template <typename Base, typename Counts>
inline constexpr bool enable_borrowed_buffer<with_counts_view<Base, Counts>> =
    enable_borrowed_buffer<Base> && enable_borrowed_buffer<Counts>;

} // namespace kamping::v2

namespace kamping::v2::views {

inline constexpr kamping::v2::adaptor<1, decltype([](auto&& r, auto&& counts) {
                                          return kamping::v2::with_counts_view(
                                              std::forward<decltype(r)>(r),
                                              std::forward<decltype(counts)>(counts)
                                          );
                                      })>
    with_counts{};

} // namespace kamping::v2::views
