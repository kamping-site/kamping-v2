// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <vector>

#include <mpi.h>

#include "kamping/v2/collectives/allgather.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/handle.hpp"
#include "kamping/v2/kassert.hpp"
#include "kamping/v2/sentinels.hpp"

/// @file
/// dstl::distribution — a monotone partition of a globally-indexed contiguous range across a
/// communicator. Given each rank's local element count (or an explicit, already globally-agreed
/// offset table), it answers purely local queries — which rank owns global index i, and local <->
/// global index conversion — in O(1) when every rank's local size is equal, or O(log p) via binary
/// search otherwise. It is a pure value type: once built, it stores no communicator, so it composes
/// freely with dstl::redistribute, dstl::alltoallv, and anywhere else "who owns global index i" is
/// needed.

namespace dstl {

/// @brief A monotone partition of a globally-indexed contiguous range across a communicator.
class distribution {
public:
    /// @brief Builds the distribution by allgathering `local_size` from every rank (one collective).
    template <kamping::v2::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
    explicit distribution(std::size_t local_size, Comm const& comm = MPI_COMM_WORLD)
        : _offset(static_cast<std::size_t>(kamping::v2::comm_view{kamping::v2::handle(comm)}.size()) + 1) {
        kamping::v2::comm_view const cv{kamping::v2::handle(comm)};
        _offset[static_cast<std::size_t>(cv.rank())] = local_size;
        kamping::v2::allgather(kamping::v2::inplace, _offset | std::views::take(cv.size()), cv);
        std::exclusive_scan(_offset.begin(), _offset.end(), _offset.begin(), std::size_t{0});
        _local_size         = local_size;
        _uniform_local_size = is_uniform();
    }

    /// @brief Builds the distribution from an explicit, already globally-agreed offset table
    /// (size comm.size() + 1, monotonically increasing). No communication is performed.
    template <std::ranges::forward_range Range, kamping::v2::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
        requires std::ranges::sized_range<Range> && std::convertible_to<std::ranges::range_value_t<Range>, std::size_t>
    distribution(Range&& offsets, Comm const& comm = MPI_COMM_WORLD)
        : _offset(std::ranges::begin(offsets), std::ranges::end(offsets)) {
        kamping::v2::comm_view const cv{kamping::v2::handle(comm)};
        KAMPING_V2_ASSERT(
            _offset.size() == static_cast<std::size_t>(cv.size()) + 1,
            "offsets size must be comm.size() + 1"
        );
        KAMPING_V2_ASSERT(std::ranges::is_sorted(_offset), "offsets must be monotonically increasing");
        _local_size = _offset[static_cast<std::size_t>(cv.rank()) + 1] - _offset[static_cast<std::size_t>(cv.rank())];
        _uniform_local_size = is_uniform();
    }

    /// @brief Rank owning global index `idx`. O(1) when local sizes are uniform, else binary search.
    [[nodiscard]] auto get_owner(std::size_t idx) const -> int {
        KAMPING_V2_ASSERT(idx < global_size(), "idx out of bounds");
        if (_uniform_local_size && _local_size > 0) {
            return static_cast<int>(idx / _local_size);
        }
        auto it = std::ranges::upper_bound(_offset, idx);
        KAMPING_V2_ASSERT(it != _offset.begin() && it != _offset.end(), "idx out of bounds");
        return static_cast<int>(std::distance(_offset.begin(), it)) - 1;
    }

    /// @brief Lazy view of each rank's local size, in rank order.
    [[nodiscard]] auto counts() const {
        return std::views::iota(0, size()) | std::views::transform([this](int rank) { return local_size(rank); });
    }

    /// @brief Lazy view of each rank's displacement (== index_range_begin(rank)), in rank order.
    /// Monotonically increasing by construction — pairs with counts() to back a variadic buffer.
    [[nodiscard]] auto displs() const {
        return std::views::iota(0, size())
               | std::views::transform([this](int rank) { return index_range_begin(rank); });
    }

    /// @brief First global index owned by `rank`.
    [[nodiscard]] auto index_range_begin(int rank) const -> std::size_t {
        return _offset[static_cast<std::size_t>(rank)];
    }

    /// @brief One past the last global index owned by `rank`.
    [[nodiscard]] auto index_range_end(int rank) const -> std::size_t {
        return _offset[static_cast<std::size_t>(rank) + 1];
    }

    /// @brief Whether global index `idx` is owned by `rank`.
    [[nodiscard]] auto is_local(std::size_t idx, int rank) const -> bool {
        return idx >= index_range_begin(rank) && idx < index_range_end(rank);
    }

    /// @brief Global -> local index conversion for an index owned by `rank`.
    [[nodiscard]] auto global_to_local(std::size_t global_idx, int rank) const -> std::size_t {
        KAMPING_V2_ASSERT(is_local(global_idx, rank), "global_idx is not owned by rank");
        return global_idx - index_range_begin(rank);
    }

    /// @brief Local -> global index conversion for `rank`.
    [[nodiscard]] auto local_to_global(std::size_t local_idx, int rank) const -> std::size_t {
        KAMPING_V2_ASSERT(local_idx < local_size(rank), "local_idx out of range for rank");
        return local_idx + index_range_begin(rank);
    }

    /// @brief Total element count across all ranks.
    [[nodiscard]] auto global_size() const -> std::size_t {
        return _offset.back();
    }

    /// @brief Element count owned by `rank`.
    [[nodiscard]] auto local_size(int rank) const -> std::size_t {
        return index_range_end(rank) - index_range_begin(rank);
    }

    /// @brief Number of ranks the distribution spans.
    [[nodiscard]] auto size() const -> int {
        return static_cast<int>(_offset.size()) - 1;
    }

    /// @brief Local indices `[0, local_size(rank))`.
    [[nodiscard]] auto local_indices(int rank) const -> std::ranges::iota_view<std::size_t, std::size_t> {
        return std::views::iota(std::size_t{0}, local_size(rank));
    }

    /// @brief Global indices `[index_range_begin(rank), index_range_end(rank))`.
    [[nodiscard]] auto global_indices(int rank) const -> std::ranges::iota_view<std::size_t, std::size_t> {
        return std::views::iota(index_range_begin(rank), index_range_end(rank));
    }

private:
    [[nodiscard]] auto is_uniform() const -> bool {
        for (std::size_t r = 0; r + 1 < _offset.size(); ++r) {
            if (_offset[r + 1] - _offset[r] != _local_size) {
                return false;
            }
        }
        return true;
    }

    std::vector<std::size_t> _offset;
    std::size_t              _local_size{};
    bool                     _uniform_local_size{};
};

} // namespace dstl
