// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once
#include <ranges>
#include <utility>

#include <mpi.h>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

/// Overrides the MPI datatype (`mpi_type()`) of the wrapped buffer.
template <typename Base>
class with_type_view : public view_interface<with_type_view<Base>> {
    Base         _base;
    MPI_Datatype _type;

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
    with_type_view(R&& base, MPI_Datatype type) : _base(kamping::v2::all(std::forward<R>(base))),
                                                  _type(type) {}

    constexpr auto mpi_type() const {
        return _type;
    }
};

template <typename R>
with_type_view(R&&, MPI_Datatype) -> with_type_view<kamping::v2::all_t<R>>;

template <typename Base>
inline constexpr bool enable_borrowed_buffer<with_type_view<Base>> = enable_borrowed_buffer<Base>;

} // namespace kamping::v2

namespace kamping::v2::views {

/// Overrides the MPI datatype: `buf | views::with_type(dt)`.
inline constexpr kamping::v2::adaptor<1, decltype([](auto&& r, MPI_Datatype type) {
                                          return kamping::v2::with_type_view(std::forward<decltype(r)>(r), type);
                                      })>
    with_type{};

} // namespace kamping::v2::views
