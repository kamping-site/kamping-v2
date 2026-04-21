#pragma once

#include <cstddef>
#include <ranges>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

template <typename Base>
class with_size_view : public view_interface<with_size_view<Base>> {
    Base           base_;
    std::ptrdiff_t size_;

public:
    constexpr Base const& base() const& noexcept {
        return base_;
    }
    constexpr Base& base() & noexcept {
        return base_;
    }

    template <typename R>
    with_size_view(R&& base, std::ptrdiff_t size) : base_(kamping::v2::all(std::forward<R>(base))),
                                                    size_(size) {}

    constexpr std::ptrdiff_t mpi_count() const {
        return size_;
    }
};

template <typename R>
with_size_view(R&&, std::ptrdiff_t) -> with_size_view<kamping::v2::all_t<R>>;

template <typename Base>
inline constexpr bool enable_borrowed_buffer<with_size_view<Base>> = enable_borrowed_buffer<Base>;

} // namespace kamping::v2

namespace kamping::v2::views {

// Useful for non-range objects that expose mpi_ptr() but no size — compose with with_type
// to build a complete data_buffer. For limiting the element count of a range, prefer std::views::take.
inline constexpr kamping::v2::adaptor<1, decltype([](auto&& r, std::ptrdiff_t size) {
                                          return kamping::v2::with_size_view(std::forward<decltype(r)>(r), size);
                                      })>
    with_size{};

} // namespace kamping::v2::views
