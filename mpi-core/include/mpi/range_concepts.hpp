// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

/// @file
/// Element-typed range concepts shared by the handle and array argument protocols.

#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>

namespace mpi::experimental {

/// A contiguous range whose elements are `E` (read access).
template <typename R, typename E>
concept contiguous_range_of =
    std::ranges::contiguous_range<R> && std::same_as<E, std::remove_cvref_t<std::ranges::range_value_t<R>>>;

/// Writable contiguous storage of `E` that refers to the caller's memory, so that MPI can write into it: an lvalue
/// reference or a borrowed view (e.g. std::span), never a container held by value.
template <typename R, typename E>
concept output_range_of = contiguous_range_of<R, E> && std::ranges::borrowed_range<R> && requires(R& r) {
    { std::ranges::data(r) } -> std::same_as<E*>;
};

/// A sized input range of standard integers (not `bool` or a character type), e.g. a `std::vector<std::size_t>`.
template <typename R>
concept sized_integer_range =
    std::ranges::input_range<R> && std::ranges::sized_range<R> && std::integral<std::ranges::range_value_t<R>>
    && !std::same_as<std::ranges::range_value_t<R>, bool> && !std::same_as<std::ranges::range_value_t<R>, char>
    && !std::same_as<std::ranges::range_value_t<R>, wchar_t> && !std::same_as<std::ranges::range_value_t<R>, char8_t>
    && !std::same_as<std::ranges::range_value_t<R>, char16_t> && !std::same_as<std::ranges::range_value_t<R>, char32_t>;

} // namespace mpi::experimental
