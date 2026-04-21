#pragma once

#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#include "kamping/kassert/kassert.hpp"
#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

template <typename Base, mpi::experimental::count_range Displs, bool resize = false>
    requires mpi::experimental::has_mpi_counts<Base> && std::ranges::output_range<Displs, int>
             && (!resize || has_resize<Displs> || has_mpi_resize_for_receive<Displs>)
class auto_displs_view : public view_interface<auto_displs_view<Base, Displs>> {
    Base           base_;
    mutable Displs displs_;
    mutable bool   needs_to_compute_displs_ = true;

public:
    constexpr Base const& base() const& noexcept {
        return base_;
    }
    constexpr Base& base() & noexcept {
        return base_;
    }

    template <typename R, typename C>
    auto_displs_view(R&& base, C&& displs)
        : base_(kamping::v2::all(std::forward<R>(base))),
          displs_(kamping::v2::all(std::forward<C>(displs))) {}

    template <typename R, typename C>
    auto_displs_view(kamping::v2::resize_t, R&& base, C&& displs)
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

    /// Displacements are always computed via exclusive_scan — monotonically non-decreasing.
    constexpr bool displs_monotonic() const {
        return true;
    }

    /// Invalidates the cached displacements so they will be recomputed on the
    /// next mpi_displs() call. Called automatically when commit_counts() propagates
    /// through from an inner auto_counts_view.
    void commit_counts() {
        needs_to_compute_displs_ = true;
    }

    std::span<int const> mpi_displs() const {
        if (needs_to_compute_displs_) {
            auto&& counts = mpi::experimental::counts(base());
            if constexpr (resize) {
                if (std::ranges::size(displs_) < std::ranges::size(counts)) {
                    kamping::v2::resize_for_receive(displs_, static_cast<std::ptrdiff_t>(std::ranges::size(counts)));
                }
            }
            KAMPING_ASSERT(std::ranges::size(displs_) >= std::ranges::size(counts));
            std::exclusive_scan(std::ranges::begin(counts), std::ranges::end(counts), std::ranges::begin(displs_), 0);
            needs_to_compute_displs_ = false;
        }
        return {displs_};
    }
};

template <typename R, typename C>
auto_displs_view(R&&, C&&) -> auto_displs_view<kamping::v2::all_t<R>, kamping::v2::all_t<C>>;

template <typename R, typename C>
auto_displs_view(kamping::v2::resize_t, R&&, C&&)
    -> auto_displs_view<kamping::v2::all_t<R>, kamping::v2::all_t<C>, true>;

template <typename Base, typename Displs>
inline constexpr bool enable_borrowed_buffer<auto_displs_view<Base, Displs>> =
    enable_borrowed_buffer<Base> && enable_borrowed_buffer<Displs>;

} // namespace kamping::v2

namespace kamping::v2::views {

// 0-arg: owned Container (default std::vector<int>), auto-resized on mpi_displs().
template <typename Container = std::vector<int>>
constexpr auto auto_displs() {
    return kamping::v2::adaptor<1, decltype([](auto&& r, auto&& displs) {
                                    return kamping::v2::auto_displs_view(
                                        kamping::v2::resize,
                                        std::forward<decltype(r)>(r),
                                        std::forward<decltype(displs)>(displs)
                                    );
                                })>{}(Container{});
}

// (container) partial or (r, container) full — no resize.
template <typename... Args>
    requires(
        sizeof...(Args) >= 1
        && !std::same_as<std::remove_cvref_t<std::tuple_element_t<0, std::tuple<Args...>>>, kamping::v2::resize_t>
    )
constexpr auto auto_displs(Args&&... args) {
    return kamping::v2::adaptor<1, decltype([](auto&& r, auto&& displs) {
                                    return kamping::v2::auto_displs_view(
                                        std::forward<decltype(r)>(r),
                                        std::forward<decltype(displs)>(displs)
                                    );
                                })>{}(std::forward<Args>(args)...);
}

// (resize, container) partial or (resize, r, container) full — with resize.
template <typename... Args>
    requires(sizeof...(Args) >= 1)
constexpr auto auto_displs(kamping::v2::resize_t, Args&&... args) {
    return kamping::v2::adaptor<1, decltype([](auto&& r, auto&& displs) {
                                    return kamping::v2::auto_displs_view(
                                        kamping::v2::resize,
                                        std::forward<decltype(r)>(r),
                                        std::forward<decltype(displs)>(displs)
                                    );
                                })>{}(std::forward<Args>(args)...);
}

} // namespace kamping::v2::views
