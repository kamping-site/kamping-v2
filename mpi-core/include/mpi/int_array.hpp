// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include "mpi/range_concepts.hpp"

/// @file
/// Array arguments that MPI reads as `int`s, but that have a typed C++ form: the distributions of
/// `type_create_darray` (`distribution`) and the logical arrays of the Cartesian topology functions (`bool`).

namespace mpi::experimental {

/// @brief An array that MPI reads as `int`s: a contiguous range of `int`, passed to MPI as it is, or a sized range of
/// `E` (e.g. `distribution` or `bool`), converted element by element.
template <typename R, typename E>
concept int_array_of = contiguous_range_of<R, int>
                       || (std::ranges::sized_range<R> && std::ranges::input_range<R>
                           && std::same_as<std::remove_cvref_t<std::ranges::range_value_t<R>>, E>);

/// @brief A logical array of MPI, e.g. the periodicity of each dimension in `cart_create`: a contiguous range of `int`
/// (nonzero is true), or a sized range of `bool`.
template <typename R>
concept flag_array = int_array_of<R, bool>;

namespace detail {
/// Calls `f` with a pointer to the elements of `r` as `int`s. A contiguous range of `int` is passed as it is; a range
/// of `E` is converted into a buffer on the stack (on the heap for more than 8 elements). Neither an enum nor a `bool`
/// may be read through an `int` pointer, and `std::vector<bool>` has no contiguous storage at all.
template <typename E, int_array_of<E> R, typename F>
decltype(auto) with_int_array(R const& r, F&& f) {
    if constexpr (contiguous_range_of<R, int>) {
        return std::forward<F>(f)(std::ranges::data(r));
    } else {
        auto const         n = static_cast<std::size_t>(std::ranges::size(r));
        std::array<int, 8> small{};
        std::vector<int>   large;
        int*               out = small.data();
        if (n > small.size()) {
            large.resize(n);
            out = large.data();
        }
        std::ranges::transform(r, out, [](E e) { return static_cast<int>(e); });
        return std::forward<F>(f)(static_cast<int const*>(out));
    }
}
} // namespace detail

} // namespace mpi::experimental
