// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstdint>
#include <type_traits>

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

/// @file
/// Rank types with neighbor arithmetic: `periodic_rank` wraps around at the ends of the communicator,
/// `nonperiodic_rank` stands for `MPI_PROC_NULL` beyond them. Both are usable wherever an MPI rank is expected.

namespace mpi::experimental {

namespace detail {
/// @return The rank of the calling process in `comm`.
/// @throws mpi_error if `MPI_Comm_rank` fails.
inline int comm_rank(MPI_Comm comm) {
    int rank = 0;
    int err  = MPI_Comm_rank(comm, &rank);
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return rank;
}

/// @return The size of `comm`.
/// @throws mpi_error if `MPI_Comm_size` fails.
inline int comm_size(MPI_Comm comm) {
    int size = 0;
    int err  = MPI_Comm_size(comm, &size);
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return size;
}

/// Tag selecting the private constructors of the rank types that take a size instead of a communicator.
struct with_size_tag {};
} // namespace detail

/// @brief A rank in a communicator of fixed size whose arithmetic wraps around at the ends, as in a ring.
///
/// `r + k` and `r - k` are the ranks `k` steps further and back along the ring, so in a communicator of size `p`,
/// `(p - 1) + 1` is `0` and `0 - 1` is `p - 1`. The value is always in `[0, size)`. The rank only stores the size of
/// the communicator, not the communicator itself.
///
/// @code
/// auto const me = periodic_rank(comm);
/// sendrecv_replace(buf, me - 1, 0, me + 1, 0, comm); // shift buf one rank to the left
/// auto const root = periodic_rank(0, comm);
/// @endcode
class periodic_rank {
    int _rank;
    int _size;

    // The arithmetic is done in 64 bits: `_rank + offset % _size` can reach `2 * size - 2`, which overflows an `int`
    // for `size > INT_MAX / 2`.
    static constexpr int wrap(std::int64_t rank, int size) {
        auto r = rank % size;
        return static_cast<int>(r < 0 ? r + size : r);
    }

    constexpr periodic_rank(int rank, int size, detail::with_size_tag) : _rank(0), _size(size) {
        if (!std::is_constant_evaluated()) {
            KAMPING_V2_ASSERT(size > 0, "a communicator has at least one rank");
        }
        _rank = wrap(rank, size);
    }

public:
    /// @brief The rank of the calling process in `comm`.
    /// @throws mpi_error if the rank or size of `comm` cannot be queried.
    template <convertible_to_mpi_handle<MPI_Comm> Comm>
    explicit periodic_rank(Comm const& comm)
        : periodic_rank(detail::comm_rank(mpi::experimental::handle(comm)), comm) {}

    /// @brief The rank `rank` in `comm`, wrapped into `[0, comm.size())`.
    /// @throws mpi_error if the size of `comm` cannot be queried.
    template <convertible_to_mpi_handle<MPI_Comm> Comm>
    periodic_rank(int rank, Comm const& comm)
        : periodic_rank(rank, detail::comm_size(mpi::experimental::handle(comm)), detail::with_size_tag{}) {}

    /// @return The rank `rank` in a communicator of size `size`, wrapped into `[0, size)`. For code that knows the
    /// size but has no communicator at hand.
    /// @pre `size > 0`.
    [[nodiscard]] static constexpr periodic_rank with_size(int rank, int size) {
        return periodic_rank(rank, size, detail::with_size_tag{});
    }

    /// @return The rank, in `[0, size)`.
    [[nodiscard]] constexpr int value() const noexcept {
        return _rank;
    }

    /// @return `value()`, for the rank protocol (`to_rank`).
    [[nodiscard]] constexpr int mpi_rank() const noexcept {
        return value();
    }

    /// @return The size of the communicator.
    [[nodiscard]] constexpr int size() const noexcept {
        return _size;
    }

    /// @brief Moves `offset` steps along the ring.
    constexpr periodic_rank& operator+=(int offset) noexcept {
        _rank = wrap(std::int64_t{_rank} + offset % _size, _size);
        return *this;
    }

    /// @brief Moves `offset` steps back along the ring.
    constexpr periodic_rank& operator-=(int offset) noexcept {
        _rank = wrap(std::int64_t{_rank} - offset % _size, _size);
        return *this;
    }

    /// @return The rank `offset` steps further along the ring.
    friend constexpr periodic_rank operator+(periodic_rank r, int offset) noexcept {
        return r += offset;
    }

    /// @return The rank `offset` steps further along the ring.
    friend constexpr periodic_rank operator+(int offset, periodic_rank r) noexcept {
        return r += offset;
    }

    /// @return The rank `offset` steps back along the ring.
    friend constexpr periodic_rank operator-(periodic_rank r, int offset) noexcept {
        return r -= offset;
    }

    /// @return Whether both ranks are the same rank in communicators of the same size.
    friend constexpr bool operator==(periodic_rank const&, periodic_rank const&) noexcept = default;
};

/// @brief A rank in a communicator of fixed size that is `MPI_PROC_NULL` beyond the ends, as on a line.
///
/// `r + k` and `r - k` are the ranks `k` steps further and back. Once the position leaves `[0, size)`, the rank stands
/// for `MPI_PROC_NULL`, so that sends to it and receives from it complete immediately without transferring data.
/// The position is mapped only when the rank is converted, so `(r + 1) - 1` is `r` again even at the last rank.
///
/// @code
/// auto const me = nonperiodic_rank(comm);
/// // send the left border to the left neighbor, receive the right halo from the right neighbor;
/// // the outermost ranks skip their missing neighbor
/// sendrecv(left_border, me - 1, 0, right_halo, me + 1, 0, comm, MPI_STATUS_IGNORE);
/// @endcode
///
/// The rank only stores the size of the communicator, not the communicator itself. A position equal to the value of
/// `MPI_PROC_NULL` has no special meaning: it is outside of `[0, size)` like any other negative position.
class nonperiodic_rank {
    // 64 bits, since the position may leave the range of `int`: `r + INT_MAX` must stay outside of `[0, size)` instead
    // of overflowing, and `(r + k) - k` must be `r` again.
    std::int64_t _position;
    int          _size;

    constexpr nonperiodic_rank(int rank, int size, detail::with_size_tag) : _position(rank), _size(size) {
        if (!std::is_constant_evaluated()) {
            KAMPING_V2_ASSERT(size > 0, "a communicator has at least one rank");
        }
    }

public:
    /// @brief The rank of the calling process in `comm`.
    /// @throws mpi_error if the rank or size of `comm` cannot be queried.
    template <convertible_to_mpi_handle<MPI_Comm> Comm>
    explicit nonperiodic_rank(Comm const& comm)
        : nonperiodic_rank(detail::comm_rank(mpi::experimental::handle(comm)), comm) {}

    /// @brief The position `rank` in `comm`. `rank` may be outside of `[0, comm.size())`.
    /// @throws mpi_error if the size of `comm` cannot be queried.
    template <convertible_to_mpi_handle<MPI_Comm> Comm>
    nonperiodic_rank(int rank, Comm const& comm)
        : nonperiodic_rank(rank, detail::comm_size(mpi::experimental::handle(comm)), detail::with_size_tag{}) {}

    /// @return The position `rank` in a communicator of size `size`. `rank` may be outside of `[0, size)`. For code
    /// that knows the size but has no communicator at hand.
    /// @pre `size > 0`.
    [[nodiscard]] static constexpr nonperiodic_rank with_size(int rank, int size) {
        return nonperiodic_rank(rank, size, detail::with_size_tag{});
    }

    /// @return Whether the position is outside of `[0, size)`, i.e. the rank stands for `MPI_PROC_NULL`.
    [[nodiscard]] constexpr bool is_proc_null() const noexcept {
        return _position < 0 || _position >= _size;
    }

    /// @return The rank if the position is in `[0, size)`, `MPI_PROC_NULL` otherwise.
    [[nodiscard]] constexpr int value() const noexcept {
        return is_proc_null() ? MPI_PROC_NULL : static_cast<int>(_position);
    }

    /// @return `value()`, for the rank protocol (`to_rank`).
    [[nodiscard]] constexpr int mpi_rank() const noexcept {
        return value();
    }

    /// @return The size of the communicator.
    [[nodiscard]] constexpr int size() const noexcept {
        return _size;
    }

    /// @brief Moves `offset` steps further.
    constexpr nonperiodic_rank& operator+=(int offset) noexcept {
        _position += offset;
        return *this;
    }

    /// @brief Moves `offset` steps back.
    constexpr nonperiodic_rank& operator-=(int offset) noexcept {
        _position -= offset;
        return *this;
    }

    /// @return The rank `offset` steps further.
    friend constexpr nonperiodic_rank operator+(nonperiodic_rank r, int offset) noexcept {
        return r += offset;
    }

    /// @return The rank `offset` steps further.
    friend constexpr nonperiodic_rank operator+(int offset, nonperiodic_rank r) noexcept {
        return r += offset;
    }

    /// @return The rank `offset` steps back.
    friend constexpr nonperiodic_rank operator-(nonperiodic_rank r, int offset) noexcept {
        return r -= offset;
    }

    /// @return Whether both stand for the same rank in communicators of the same size. All positions outside of
    /// `[0, size)` are equal, since they all stand for `MPI_PROC_NULL`.
    friend constexpr bool operator==(nonperiodic_rank const& lhs, nonperiodic_rank const& rhs) noexcept {
        return lhs._size == rhs._size && lhs.value() == rhs.value();
    }
};

} // namespace mpi::experimental
