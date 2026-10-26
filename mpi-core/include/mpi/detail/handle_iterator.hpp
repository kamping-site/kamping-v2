// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <compare>
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace mpi::experimental::detail {

/// Random-access proxy iterator over a contiguous array of `Handle` that yields a `View` by value on dereference.
///
/// Used by request_vector and status_vector so that iterating them yields request_view / status_view instead of
/// raw MPI handles. `View` must be constructible from `Handle&` or from `Handle*`.
///
/// Since dereferencing yields a prvalue, the iterator is a C++20 `random_access_iterator`, but only a legacy input
/// iterator; it is deliberately not contiguous.
template <typename Handle, typename View>
class handle_iterator {
public:
    using iterator_concept  = std::random_access_iterator_tag;
    using iterator_category = std::input_iterator_tag;
    using value_type        = View;
    using difference_type   = std::ptrdiff_t;

    handle_iterator() = default;

    explicit handle_iterator(Handle* ptr) noexcept : _ptr(ptr) {}

    /// @return Pointer to the underlying handle.
    [[nodiscard]] Handle* base() const noexcept {
        return _ptr;
    }

    View operator*() const noexcept {
        return make_view(_ptr);
    }

    View operator[](difference_type n) const noexcept {
        return make_view(_ptr + n);
    }

    handle_iterator& operator++() noexcept {
        ++_ptr;
        return *this;
    }

    handle_iterator operator++(int) noexcept {
        auto tmp = *this;
        ++_ptr;
        return tmp;
    }

    handle_iterator& operator--() noexcept {
        --_ptr;
        return *this;
    }

    handle_iterator operator--(int) noexcept {
        auto tmp = *this;
        --_ptr;
        return tmp;
    }

    handle_iterator& operator+=(difference_type n) noexcept {
        _ptr += n;
        return *this;
    }

    handle_iterator& operator-=(difference_type n) noexcept {
        _ptr -= n;
        return *this;
    }

    friend handle_iterator operator+(handle_iterator it, difference_type n) noexcept {
        return it += n;
    }

    friend handle_iterator operator+(difference_type n, handle_iterator it) noexcept {
        return it += n;
    }

    friend handle_iterator operator-(handle_iterator it, difference_type n) noexcept {
        return it -= n;
    }

    friend difference_type operator-(handle_iterator const& lhs, handle_iterator const& rhs) noexcept {
        return lhs._ptr - rhs._ptr;
    }

    friend bool operator==(handle_iterator const& lhs, handle_iterator const& rhs) noexcept = default;

    friend std::strong_ordering operator<=>(handle_iterator const& lhs, handle_iterator const& rhs) noexcept {
        return lhs._ptr <=> rhs._ptr;
    }

private:
    static View make_view(Handle* ptr) noexcept {
        if constexpr (std::is_constructible_v<View, Handle&>) {
            return View{*ptr};
        } else {
            return View{ptr};
        }
    }

    Handle* _ptr = nullptr;
};

} // namespace mpi::experimental::detail
