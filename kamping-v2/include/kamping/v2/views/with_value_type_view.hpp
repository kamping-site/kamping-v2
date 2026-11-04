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

/// Mirrors with_type_view but sets the value-type channel: mpi_value_type() (the MPI type of the
/// payload of a structured buffer, see kamping::v2::value_type()), not mpi_type(). Only consumers
/// of structured buffers (e.g. flatten_v_view) read it; mpi_type() is left to the base.
template <typename Base>
class with_value_type_view : public view_interface<with_value_type_view<Base>> {
    Base         _base;
    MPI_Datatype _value_type;

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
    with_value_type_view(R&& base, MPI_Datatype value_type)
        : _base(kamping::v2::all(std::forward<R>(base))),
          _value_type(value_type) {}

    constexpr auto mpi_value_type() const {
        return _value_type;
    }
};

template <typename R>
with_value_type_view(R&&, MPI_Datatype) -> with_value_type_view<kamping::v2::all_t<R>>;

template <typename Base>
inline constexpr bool enable_borrowed_buffer<with_value_type_view<Base>> = enable_borrowed_buffer<Base>;

} // namespace kamping::v2

namespace kamping::v2::views {

/// Sets the MPI datatype of the payload of a structured buffer: `buf | views::with_value_type(dt)`.
inline constexpr kamping::v2::adaptor<1, decltype([](auto&& r, MPI_Datatype value_type) {
                                          return kamping::v2::with_value_type_view(
                                              std::forward<decltype(r)>(r),
                                              value_type
                                          );
                                      })>
    with_value_type{};

} // namespace kamping::v2::views
