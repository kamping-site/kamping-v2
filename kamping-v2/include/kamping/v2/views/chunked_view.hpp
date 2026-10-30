// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

/// @file
/// @brief `views::chunked`: a variadic buffer as a range of per-rank chunks, the inverse of `views::flatten_v`.

#include <compare>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <span>
#include <utility>

#include <mpi.h>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/views/payload.hpp"
#include "mpi/buffer.hpp"

namespace kamping::v2 {

/// @brief A variadic buffer (data + per-rank counts + displacements) seen as a range of chunks, one per rank.
///
/// Chunk `i` is a `std::span` over `counts[i]` elements starting at `displs[i]`: it refers to the buffer's data, so the
/// buffer must outlive the chunks. This is the inverse of `views::flatten_v`, and makes a flat buffer a dense nested
/// buffer (`nested_send_buffer`) indexed by destination, e.g. as input of `dstl::sparse_alltoall`.
///
/// The value-type channel (`kamping::v2::value_type`) of the chunked view is the buffer's own MPI type, so
/// `buf | views::with_type(dt) | views::chunked` carries `dt` to consumers of nested buffers. Chunks are read-only.
///
/// @tparam Base A variadic buffer (`data_buffer_v`) with an element type (`range_value_t` or a `value_type` member).
template <typename Base>
    requires mpi::experimental::data_buffer_v<Base const&> && requires { typename mpi_element_type_t<Base>; }
class chunked_view : public std::ranges::view_interface<chunked_view<Base>> {
public:
    /// Element type of the buffer.
    using element_type = mpi_element_type_t<Base>;
    /// A chunk: the elements for one rank.
    using chunk_type = std::span<element_type const>;

    /// Random-access iterator over the chunks, yielding them by value.
    class iterator {
    public:
        using iterator_concept = std::random_access_iterator_tag;
        using value_type       = chunk_type;
        using difference_type  = std::ptrdiff_t;

        iterator() = default;
        iterator(chunked_view const* view, difference_type i) noexcept : _view(view), _i(i) {}

        [[nodiscard]] chunk_type operator*() const {
            return _view->chunk(_i);
        }
        [[nodiscard]] chunk_type operator[](difference_type n) const {
            return _view->chunk(_i + n);
        }

        iterator& operator++() noexcept {
            ++_i;
            return *this;
        }
        iterator operator++(int) noexcept {
            auto tmp = *this;
            ++_i;
            return tmp;
        }
        iterator& operator--() noexcept {
            --_i;
            return *this;
        }
        iterator operator--(int) noexcept {
            auto tmp = *this;
            --_i;
            return tmp;
        }
        iterator& operator+=(difference_type n) noexcept {
            _i += n;
            return *this;
        }
        iterator& operator-=(difference_type n) noexcept {
            _i -= n;
            return *this;
        }
        [[nodiscard]] friend iterator operator+(iterator it, difference_type n) noexcept {
            return it += n;
        }
        [[nodiscard]] friend iterator operator+(difference_type n, iterator it) noexcept {
            return it += n;
        }
        [[nodiscard]] friend iterator operator-(iterator it, difference_type n) noexcept {
            return it -= n;
        }
        [[nodiscard]] friend difference_type operator-(iterator const& a, iterator const& b) noexcept {
            return a._i - b._i;
        }
        [[nodiscard]] friend bool operator==(iterator const& a, iterator const& b) noexcept {
            return a._i == b._i;
        }
        [[nodiscard]] friend auto operator<=>(iterator const& a, iterator const& b) noexcept {
            return a._i <=> b._i;
        }

    private:
        chunked_view const* _view = nullptr;
        difference_type     _i    = 0;
    };

    template <typename R>
    explicit chunked_view(R&& base) : _base(kamping::v2::all(std::forward<R>(base))) {}

    /// @return The number of chunks, i.e. the number of ranks the buffer has counts for.
    [[nodiscard]] std::size_t size() const {
        return static_cast<std::size_t>(std::ranges::size(mpi::experimental::counts(_base)));
    }

    [[nodiscard]] iterator begin() const noexcept {
        return iterator(this, 0);
    }
    [[nodiscard]] iterator end() const {
        return iterator(this, static_cast<std::ptrdiff_t>(size()));
    }

    /// @return The chunk of rank `i`.
    [[nodiscard]] chunk_type chunk(std::ptrdiff_t i) const {
        auto const  count = std::ranges::begin(mpi::experimental::counts(_base))[i];
        auto const  displ = std::ranges::begin(mpi::experimental::displs(_base))[i];
        auto const* data  = static_cast<element_type const*>(static_cast<void const*>(mpi::experimental::ptr(_base)));
        return chunk_type(data + displ, static_cast<std::size_t>(count));
    }

    /// @return The MPI type of the buffer, as the value type of the chunks (see `kamping::v2::value_type`).
    [[nodiscard]] MPI_Datatype mpi_value_type() const {
        return mpi::experimental::type(_base);
    }

    [[nodiscard]] Base const& base() const& noexcept {
        return _base;
    }
    [[nodiscard]] Base&& base() && noexcept {
        return std::move(_base);
    }

private:
    Base _base;
};

template <typename R>
chunked_view(R&&) -> chunked_view<kamping::v2::all_t<R>>;

} // namespace kamping::v2

namespace kamping::v2::views {

/// Splits a variadic buffer into its per-rank chunks: `buf | views::chunked` is a range of `std::span`s, chunk `i`
/// holding the elements for rank `i`. The inverse of `views::flatten_v`.
inline constexpr struct chunked_fn : kamping::v2::adaptor_closure<chunked_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return kamping::v2::chunked_view(std::forward<R>(r));
    }
} chunked{};

} // namespace kamping::v2::views
