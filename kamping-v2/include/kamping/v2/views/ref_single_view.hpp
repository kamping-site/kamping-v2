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
/// Satisfies contiguous_range and sized_range (size is always 1). MPI buffer
/// protocol (mpi_data, mpi_count, mpi_type) is handled automatically by the
/// CPO dispatch in ranges.hpp for contiguous ranges of builtin MPI types.
///
/// Typical usage:
///   int val = 42;
///   kamping::v2::bcast(ref_single(val));   // broadcasts val in-place
template <typename T>
    requires std::is_object_v<T>
class ref_single_view : public std::ranges::view_interface<ref_single_view<T>> {
    T* ptr_;

public:
    constexpr explicit ref_single_view(T& val) noexcept : ptr_(std::addressof(val)) {}

    constexpr T* begin() noexcept {
        return ptr_;
    }
    constexpr T const* begin() const noexcept {
        return ptr_;
    }
    constexpr T* end() noexcept {
        return ptr_ + 1;
    }
    constexpr T const* end() const noexcept {
        return ptr_ + 1;
    }

    static constexpr bool empty() noexcept {
        return false;
    }
    static constexpr std::size_t size() noexcept {
        return 1;
    }

    constexpr T* data() noexcept {
        return ptr_;
    }
    constexpr T const* data() const noexcept {
        return ptr_;
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
