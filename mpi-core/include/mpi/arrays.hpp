// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

/// @file
/// Array arguments of MPI functions: the request array that determines the count, and output arrays that MPI fills
/// with up to that many elements (statuses, indices, ...).
///
/// A raw pointer is never accepted as an array, since it carries no length: wrap it as `std::span(ptr, n)`. Special
/// pointer values such as `MPI_STATUSES_IGNORE` are sentinel types (see mpi/sentinels.hpp), as for buffers.

#pragma once

#include <cstddef>
#include <ranges>
#include <type_traits>
#include <utility>

#include "kamping/v2/kassert.hpp"
#include "mpi/handle.hpp"
#include "mpi/range_concepts.hpp"

namespace mpi::experimental {

/// The request array of a multi-request call (`waitall`, `waitany`, `waitsome`, ...): a writable contiguous range
/// of `MPI_Request` obtained through handle_range(). It must be sized, since its size is the request count.
template <typename T>
concept request_range = convertible_to_mpi_handle_range<T, MPI_Request>
                        && std::ranges::sized_range<decltype(mpi::experimental::handle_range(std::declval<T&>()))>;

/// An output array of `E`. Handle elements are obtained through handle_range(), so that types such as
/// `status_vector` can expose their storage; other element types (e.g. `int` indices) must be a plain contiguous
/// range. A sized array is checked to hold the required number of elements; an unsized one is trusted.
template <typename T, typename E>
concept output_array_of =
    (builtin_handle<E> && convertible_to_mpi_handle_range<T, E>) || (!builtin_handle<E> && output_range_of<T&, E>);

/// A sentinel type standing in for an ignorable output array of `E`: not a range, with a static `mpi_ptr()`
/// returning the special pointer value (e.g. `statuses_ignore` → `MPI_STATUSES_IGNORE`).
template <typename T, typename E>
concept output_array_sentinel_of = !std::ranges::range<std::remove_cvref_t<T>> && requires {
    { std::remove_cvref_t<T>::mpi_ptr() } -> std::same_as<E*>;
};

/// An output array of `E` that MPI allows to be ignored: an `output_array_of<E>` or a matching sentinel.
template <typename T, typename E>
concept output_array_or_ignore_of = output_array_of<T, E> || output_array_sentinel_of<T, E>;

namespace detail {
template <typename E, typename R>
E* checked_output_ptr(R&& range, [[maybe_unused]] std::size_t required) {
    if constexpr (std::ranges::sized_range<R>) {
        KAMPING_V2_ASSERT(
            static_cast<std::size_t>(std::ranges::size(range)) >= required,
            "output array has fewer entries than required"
        );
    }
    return std::ranges::data(range);
}

/// Returns the pointer to pass to MPI for an output array that must hold at least `required` elements.
template <typename E, typename T>
    requires output_array_or_ignore_of<T, E>
E* output_array_ptr(T& array, [[maybe_unused]] std::size_t required) {
    if constexpr (output_array_sentinel_of<T, E>) {
        return std::remove_cvref_t<T>::mpi_ptr();
    } else if constexpr (builtin_handle<E>) {
        return checked_output_ptr<E>(mpi::experimental::handle_range(array), required);
    } else {
        return checked_output_ptr<E>(array, required);
    }
}
} // namespace detail

} // namespace mpi::experimental
