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
#include "kamping/v2/kassert.hpp"
#include "kamping/v2/sentinels.hpp"
#include "mpi/handle.hpp"

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
    template <mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
    explicit distribution(std::size_t local_size, Comm const& comm = MPI_COMM_WORLD)
        : offset_(static_cast<std::size_t>(mpi::experimental::comm_view{mpi::experimental::handle(comm)}.size()) + 1) {
        kamping::v2::comm_view const cv{mpi::experimental::handle(comm)};
        offset_[static_cast<std::size_t>(cv.rank())] = local_size;
        kamping::v2::allgather(kamping::v2::inplace, offset_ | std::views::take(cv.size()), cv);
        std::exclusive_scan(offset_.begin(), offset_.end(), offset_.begin(), std::size_t{0});
        local_size_         = local_size;
        uniform_local_size_ = is_uniform();
    }

    /// @brief Builds the distribution from an explicit, already globally-agreed offset table
    /// (size comm.size() + 1, monotonically increasing). No communication is performed.
    template <std::ranges::forward_range Range, mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
        requires std::ranges::sized_range<Range> && std::convertible_to<std::ranges::range_value_t<Range>, std::size_t>
    distribution(Range&& offsets, Comm const& comm = MPI_COMM_WORLD)
        : offset_(std::ranges::begin(offsets), std::ranges::end(offsets)) {
        kamping::v2::comm_view const cv{mpi::experimental::handle(comm)};
        KAMPING_V2_ASSERT(
            offset_.size() == static_cast<std::size_t>(cv.size()) + 1,
            "offsets size must be comm.size() + 1"
        );
        KAMPING_V2_ASSERT(std::ranges::is_sorted(offset_), "offsets must be monotonically increasing");
        local_size_ = offset_[static_cast<std::size_t>(cv.rank()) + 1] - offset_[static_cast<std::size_t>(cv.rank())];
        uniform_local_size_ = is_uniform();
    }

    /// @brief Rank owning global index `idx`. O(1) when local sizes are uniform, else binary search.
    [[nodiscard]] auto get_owner(std::size_t idx) const -> int {
        KAMPING_V2_ASSERT(idx < global_size(), "idx out of bounds");
        if (uniform_local_size_ && local_size_ > 0) {
            return static_cast<int>(idx / local_size_);
        }
        auto it = std::ranges::upper_bound(offset_, idx);
        KAMPING_V2_ASSERT(it != offset_.begin() && it != offset_.end(), "idx out of bounds");
        return static_cast<int>(std::distance(offset_.begin(), it)) - 1;
    }

    /// @brief Lazy view of each rank's local size, in rank order.
    [[nodiscard]] auto counts() const {
        return std::views::iota(0, size()) | std::views::transform([this](int rank) { return local_size(rank); });
    }

    /// @brief First global index owned by `rank`.
    [[nodiscard]] auto index_range_begin(int rank) const -> std::size_t {
        return offset_[static_cast<std::size_t>(rank)];
    }

    /// @brief One past the last global index owned by `rank`.
    [[nodiscard]] auto index_range_end(int rank) const -> std::size_t {
        return offset_[static_cast<std::size_t>(rank) + 1];
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
        return offset_.back();
    }

    /// @brief Element count owned by `rank`.
    [[nodiscard]] auto local_size(int rank) const -> std::size_t {
        return index_range_end(rank) - index_range_begin(rank);
    }

    /// @brief Number of ranks the distribution spans.
    [[nodiscard]] auto size() const -> int {
        return static_cast<int>(offset_.size()) - 1;
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
        for (std::size_t r = 0; r + 1 < offset_.size(); ++r) {
            if (offset_[r + 1] - offset_[r] != local_size_) {
                return false;
            }
        }
        return true;
    }

    std::vector<std::size_t> offset_;
    std::size_t              local_size_{};
    bool                     uniform_local_size_{};
};

} // namespace dstl
