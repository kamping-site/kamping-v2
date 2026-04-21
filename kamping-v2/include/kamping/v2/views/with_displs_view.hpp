#pragma once

#include <ranges>
#include <span>
#include <tuple>
#include <utility>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

template <typename Base, mpi::experimental::count_range Displs, bool monotonic = false>
class with_displs_view : public view_interface<with_displs_view<Base, Displs, monotonic>> {
    Base   base_;
    Displs displs_;

public:
    constexpr Base const& base() const& noexcept {
        return base_;
    }
    constexpr Base& base() & noexcept {
        return base_;
    }

    template <typename R, typename C>
    with_displs_view(R&& base, C&& displs)
        : base_(kamping::v2::all(std::forward<R>(base))),
          displs_(kamping::v2::all(std::forward<C>(displs))) {}

    /// Tag-dispatch constructor: marks displacements as monotonically non-decreasing,
    /// enabling the O(1) fast path in resize_v_view.
    template <typename R, typename C>
    with_displs_view(kamping::v2::monotonic_t, R&& base, C&& displs)
        : base_(kamping::v2::all(std::forward<R>(base))),
          displs_(kamping::v2::all(std::forward<C>(displs))) {}

    constexpr Displs const& displs() const& {
        return displs_;
    }
    constexpr Displs& displs() & {
        return displs_;
    }
    constexpr Displs&& displs() && {
        return std::move(displs_);
    }

    constexpr std::pair<Base, Displs> extract() && {
        return {std::move(base_), std::move(displs_)};
    }

    std::span<int const> mpi_displs() const {
        return {displs_};
    }

    /// User-declared displacements are monotonic when constructed with the monotonic tag.
    constexpr bool displs_monotonic() const
        requires(monotonic)
    {
        return true;
    }
};

template <typename R, typename C>
with_displs_view(R&&, C&&) -> with_displs_view<kamping::v2::all_t<R>, kamping::v2::all_t<C>>;

template <typename R, typename C>
with_displs_view(kamping::v2::monotonic_t, R&&, C&&)
    -> with_displs_view<kamping::v2::all_t<R>, kamping::v2::all_t<C>, true>;

template <typename Base, typename Displs, bool monotonic>
inline constexpr bool enable_borrowed_buffer<with_displs_view<Base, Displs, monotonic>> =
    enable_borrowed_buffer<Base> && enable_borrowed_buffer<Displs>;

} // namespace kamping::v2

namespace kamping::v2::views {

inline constexpr struct with_displs_fn {
    // Partial / full (non-monotonic): with_displs(displs)  or  with_displs(r, displs)
    template <typename... Args>
        requires(
            sizeof...(Args) >= 1
            && !std::
                   same_as<std::remove_cvref_t<std::tuple_element_t<0, std::tuple<Args...>>>, kamping::v2::monotonic_t>
        )
    constexpr auto operator()(Args&&... args) const {
        return kamping::v2::adaptor<1, decltype([](auto&& r, auto&& d) {
                                        return kamping::v2::with_displs_view(
                                            std::forward<decltype(r)>(r),
                                            std::forward<decltype(d)>(d)
                                        );
                                    })>{}(std::forward<Args>(args)...);
    }

    // Partial (monotonic): with_displs(monotonic, displs)
    template <typename C>
    constexpr auto operator()(kamping::v2::monotonic_t, C&& displs) const {
        return kamping::v2::adaptor<1, decltype([](auto&& r, auto&& d) {
                                        return kamping::v2::with_displs_view(
                                            kamping::v2::monotonic,
                                            std::forward<decltype(r)>(r),
                                            std::forward<decltype(d)>(d)
                                        );
                                    })>{}(std::forward<C>(displs));
    }

    // Full (monotonic): with_displs(monotonic, r, displs)
    template <typename R, typename C>
    constexpr auto operator()(kamping::v2::monotonic_t, R&& r, C&& displs) const {
        return kamping::v2::with_displs_view(kamping::v2::monotonic, std::forward<R>(r), std::forward<C>(displs));
    }
} with_displs{};

} // namespace kamping::v2::views
