#pragma once
#include <ranges>

#include <mpi.h>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

template <typename Base>
class with_type_view : public view_interface<with_type_view<Base>> {
    Base         base_;
    MPI_Datatype type_;

public:
    constexpr Base const& base() const& noexcept {
        return base_;
    }
    constexpr Base& base() & noexcept {
        return base_;
    }

    template <typename R>
    with_type_view(R&& base, MPI_Datatype type) : base_(kamping::v2::all(std::forward<R>(base))),
                                                  type_(type) {}

    constexpr auto mpi_type() const {
        return type_;
    }
};

template <typename R>
with_type_view(R&&, MPI_Datatype) -> with_type_view<kamping::v2::all_t<R>>;

template <typename Base>
inline constexpr bool enable_borrowed_buffer<with_type_view<Base>> = enable_borrowed_buffer<Base>;

} // namespace kamping::v2

namespace kamping::v2::views {

inline constexpr kamping::v2::adaptor<1, decltype([](auto&& r, MPI_Datatype type) {
                                          return kamping::v2::with_type_view(std::forward<decltype(r)>(r), type);
                                      })>
    with_type{};

} // namespace kamping::v2::views
