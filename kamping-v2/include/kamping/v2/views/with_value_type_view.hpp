// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once
#include <ranges>

#include <mpi.h>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping::v2 {

/// Mirrors with_type_view but writes the other channel: mpi_value_type() (the payload/value
/// slot's MPI type), not mpi_type(). Forwards the range (begin/end) and leaves mpi_type() to
/// the base — for a raw structured range that has none, so a normal send/recv still cannot
/// see this annotation, only structured consumers (flatten_v_view, dstl::request_reply) that
/// read kamping::v2::value_type() do.
template <typename Base>
class with_value_type_view : public view_interface<with_value_type_view<Base>> {
    Base         base_;
    MPI_Datatype value_type_;

public:
    constexpr Base const& base() const& noexcept {
        return base_;
    }
    constexpr Base& base() & noexcept {
        return base_;
    }

    template <typename R>
    with_value_type_view(R&& base, MPI_Datatype value_type)
        : base_(kamping::v2::all(std::forward<R>(base))),
          value_type_(value_type) {}

    constexpr auto mpi_value_type() const {
        return value_type_;
    }
};

template <typename R>
with_value_type_view(R&&, MPI_Datatype) -> with_value_type_view<kamping::v2::all_t<R>>;

template <typename Base>
inline constexpr bool enable_borrowed_buffer<with_value_type_view<Base>> = enable_borrowed_buffer<Base>;

} // namespace kamping::v2

namespace kamping::v2::views {

inline constexpr kamping::v2::adaptor<1, decltype([](auto&& r, MPI_Datatype value_type) {
                                          return kamping::v2::with_value_type_view(
                                              std::forward<decltype(r)>(r),
                                              value_type
                                          );
                                      })>
    with_value_type{};

} // namespace kamping::v2::views
