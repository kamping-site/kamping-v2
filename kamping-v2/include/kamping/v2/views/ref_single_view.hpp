// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <memory>
#include <ranges>
#include <type_traits>

#include "kamping/v2/views/concepts.hpp"

namespace kamping::v2 {

/// Non-owning view over a single element, modeled after std::views::single but
/// capturing by lvalue reference instead of by value.
///
/// Satisfies contiguous_range and sized_range (size is always 1), so the buffer
/// accessors (count, ptr, type) work through the standard range fallback for
/// builtin MPI types.
///
/// Typical usage:
///   int val = 42;
///   kamping::v2::bcast(ref_single(val));   // broadcasts val in-place
template <typename T>
    requires std::is_object_v<T>
class ref_single_view : public std::ranges::view_interface<ref_single_view<T>> {
    T* _ptr;

public:
    constexpr explicit ref_single_view(T& val) noexcept : _ptr(std::addressof(val)) {}

    constexpr T* begin() noexcept {
        return _ptr;
    }
    constexpr T const* begin() const noexcept {
        return _ptr;
    }
    constexpr T* end() noexcept {
        return _ptr + 1;
    }
    constexpr T const* end() const noexcept {
        return _ptr + 1;
    }

    static constexpr bool empty() noexcept {
        return false;
    }
    static constexpr std::size_t size() noexcept {
        return 1;
    }

    constexpr T* data() noexcept {
        return _ptr;
    }
    constexpr T const* data() const noexcept {
        return _ptr;
    }
};

template <typename T>
inline constexpr bool enable_borrowed_buffer<ref_single_view<T>> = true;

} // namespace kamping::v2

template <typename T>
inline constexpr bool std::ranges::enable_borrowed_range<kamping::v2::ref_single_view<T>> = true;

namespace kamping::v2::views {

/// Wraps a single lvalue element as a size-1 view, capturing by reference.
///   int val = 42;
///   auto buf = ref_single(val);                  // ref_single_view<int>
///   auto buf = ref_single(std::as_const(val));   // ref_single_view<int const>
template <typename T>
constexpr auto ref_single(T& val) noexcept {
    return kamping::v2::ref_single_view<T>(val);
}

} // namespace kamping::v2::views
