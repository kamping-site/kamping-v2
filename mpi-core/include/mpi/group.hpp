// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <array>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

// ── GroupEquality ─────────────────────────────────────────────────────────────

/// @brief Result of `MPI_Group_compare`.
///
/// No ordering operator — the values are not comparable by magnitude.
enum class GroupEquality {
    Identical, ///< Same group object (same processes in same order).
    Similar,   ///< Same processes but in different order.
    Unequal,   ///< Different process sets.
};

// ── Forward declarations ──────────────────────────────────────────────────────

class group_view;
class group;

// ── group_accessors ───────────────────────────────────────────────────────────

/// @brief CRTP mixin providing read-only accessors for any group wrapper.
///
/// @tparam Derived Must implement `MPI_Group mpi_handle() const noexcept`.
template <typename Derived>
class group_accessors {
    [[nodiscard]] MPI_Group grp() const noexcept {
        return static_cast<Derived const*>(this)->mpi_handle();
    }

public:
    /// @return Number of processes in the group.
    [[nodiscard]] int size() const {
        int s = 0;
        MPI_Group_size(grp(), &s);
        return s;
    }

    /// @return Rank of the calling process in this group, or `std::nullopt` if
    ///         the calling process is not a member (`MPI_UNDEFINED`).
    [[nodiscard]] std::optional<int> rank() const {
        int r = MPI_UNDEFINED;
        MPI_Group_rank(grp(), &r);
        if (r == MPI_UNDEFINED) {
            return std::nullopt;
        }
        return r;
    }

    /// @return `true` if the calling process is a member of this group.
    [[nodiscard]] bool contains_self() const {
        return rank().has_value();
    }

    /// @brief Compare two groups.
    /// @return `GroupEquality::Identical` if same object, `Similar` if same
    ///         processes in different order, `Unequal` otherwise.
    [[nodiscard]] GroupEquality compare(group_view other) const;

    // ── Rank translation ──────────────────────────────────────────────────────

    /// @brief Translate a single rank from this group to `other`.
    ///
    /// Calls `MPI_Group_translate_ranks` with n=1.
    /// @return The corresponding rank in `other`, or `std::nullopt` if the
    ///         process is not a member of `other` (`MPI_UNDEFINED`).
    /// @throws mpi_error on failure.
    [[nodiscard]] std::optional<int> translate_rank(int r, group_view other) const;

    /// @brief Translate multiple ranks from this group to `other`.
    ///
    /// Each entry of the returned vector corresponds to the same-indexed rank
    /// in `ranks`. An entry is `std::nullopt` when the process is not a member
    /// of `other` (`MPI_UNDEFINED`).
    /// @throws mpi_error on failure.
    [[nodiscard]] std::vector<std::optional<int>> translate_ranks(std::span<int const> ranks, group_view other) const;

    // ── Subgroup selection ────────────────────────────────────────────────────

    /// @brief Create a subgroup containing exactly the listed ranks (in order).
    ///
    /// Calls `MPI_Group_incl`. The caller owns the returned group.
    /// @throws mpi_error on failure.
    [[nodiscard]] group include(std::span<int const> ranks) const;

    /// @brief Create a subgroup that excludes the listed ranks.
    ///
    /// Calls `MPI_Group_excl`. The caller owns the returned group.
    /// @throws mpi_error on failure.
    [[nodiscard]] group exclude(std::span<int const> ranks) const;

    /// @brief Create a subgroup from a set of (first, last, stride) rank ranges.
    ///
    /// Each element of `ranges` is a `{first, last, stride}` triplet.
    /// Calls `MPI_Group_range_incl`. The caller owns the returned group.
    /// @throws mpi_error on failure.
    [[nodiscard]] group include_ranges(std::span<std::array<int, 3>> ranges) const;

    /// @brief Create a subgroup by excluding a set of (first, last, stride) rank ranges.
    ///
    /// Calls `MPI_Group_range_excl`. The caller owns the returned group.
    /// @throws mpi_error on failure.
    [[nodiscard]] group exclude_ranges(std::span<std::array<int, 3>> ranges) const;
};

// ── group_view ────────────────────────────────────────────────────────────────

/// @brief Non-owning wrapper around an `MPI_Group`.
///
/// Satisfies `convertible_to_mpi_handle<MPI_Group>`. Does not free the group
/// on destruction — use the owning `group` for that.
class group_view : public group_accessors<group_view> {
public:
    /// @brief Construct from a raw `MPI_Group`. The group must outlive this view.
    explicit group_view(MPI_Group g) noexcept : _group(g) {}

    /// @return The underlying `MPI_Group` (for `handle()` dispatch).
    [[nodiscard]] MPI_Group mpi_handle() const noexcept { return _group; }

private:
    MPI_Group _group;
};

// ── group ─────────────────────────────────────────────────────────────────────

/// @brief Owning RAII wrapper for `MPI_Group`.
///
/// Move-only. Calls `MPI_Group_free` on destruction unless the stored handle
/// is `MPI_GROUP_EMPTY` (the predefined empty group, which must not be freed).
///
/// Satisfies `convertible_to_mpi_handle<MPI_Group>`.
class group : public group_accessors<group> {
public:
    /// @brief Default-construct (holds `MPI_GROUP_EMPTY`).
    group() noexcept = default;

    group(group const&)            = delete;
    group& operator=(group const&) = delete;

    /// @brief Move constructor — transfers ownership; moved-from holds `MPI_GROUP_EMPTY`.
    group(group&& o) noexcept : _group(std::exchange(o._group, MPI_GROUP_EMPTY)) {}

    /// @brief Move assignment — frees the current handle then transfers ownership.
    group& operator=(group&& o) noexcept {
        if (this != &o) {
            free_if_valid();
            _group = std::exchange(o._group, MPI_GROUP_EMPTY);
        }
        return *this;
    }

    /// @brief Free the group (unless it is `MPI_GROUP_EMPTY` or moved from).
    ~group() noexcept {
        free_if_valid();
    }

    /// @brief Return an owning wrapper around `MPI_GROUP_EMPTY`.
    ///
    /// The predefined empty group is never freed — `free_if_valid` skips it.
    [[nodiscard]] static group empty() noexcept {
        return group(MPI_GROUP_EMPTY, adopt_t{});
    }

    /// @brief Relinquish ownership; returns the raw `MPI_Group` handle.
    ///
    /// Leaves `*this` as `MPI_GROUP_EMPTY`. The caller is responsible for calling
    /// `MPI_Group_free` on the returned handle.
    [[nodiscard]] MPI_Group release() noexcept {
        return std::exchange(_group, MPI_GROUP_EMPTY);
    }

    /// @brief Adopt an already-created `MPI_Group` handle (takes ownership).
    ///
    /// Use when you have called an MPI routine that returns an `MPI_Group` and
    /// you want RAII management. Do not pass `MPI_GROUP_EMPTY` (it is never
    /// freed anyway, but using `group::empty()` is clearer).
    [[nodiscard]] static group from_native(MPI_Group g) noexcept {
        return group(g, adopt_t{});
    }

    /// @brief Implicit conversion to a non-owning view.
    ///
    /// Allows passing a `group` wherever a `group_view` is expected without
    /// an explicit cast. The view borrows; it does not extend lifetime.
    operator group_view() const noexcept { return group_view{_group}; }

    /// @return The underlying `MPI_Group` (for `handle()` dispatch).
    [[nodiscard]] MPI_Group mpi_handle() const noexcept { return _group; }

private:
    struct adopt_t {};

    group(MPI_Group g, adopt_t) noexcept : _group(g) {}

    void free_if_valid() noexcept {
        if (_group != MPI_GROUP_EMPTY) {
            MPI_Group_free(&_group);
            _group = MPI_GROUP_EMPTY;
        }
    }

    MPI_Group _group = MPI_GROUP_EMPTY;
};

// ── group_accessors out-of-line definitions (need both group_view and group) ───

template <typename Derived>
GroupEquality group_accessors<Derived>::compare(group_view other) const {
    int result = MPI_IDENT;
    MPI_Group_compare(grp(), other.mpi_handle(), &result);
    switch (result) {
        case MPI_IDENT:   return GroupEquality::Identical;
        case MPI_SIMILAR: return GroupEquality::Similar;
        default:          return GroupEquality::Unequal;
    }
}

template <typename Derived>
std::optional<int> group_accessors<Derived>::translate_rank(int r, group_view other) const {
    int out = MPI_UNDEFINED;
    int err = MPI_Group_translate_ranks(grp(), 1, &r, other.mpi_handle(), &out);
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    if (out == MPI_UNDEFINED) {
        return std::nullopt;
    }
    return out;
}

template <typename Derived>
std::vector<std::optional<int>> group_accessors<Derived>::translate_ranks(
    std::span<int const> ranks, group_view other
) const {
    std::vector<int> raw(ranks.size(), MPI_UNDEFINED);
    int              err = MPI_Group_translate_ranks(
        grp(), static_cast<int>(ranks.size()), ranks.data(), other.mpi_handle(), raw.data()
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    std::vector<std::optional<int>> result;
    result.reserve(ranks.size());
    for (int v : raw) {
        result.push_back(v == MPI_UNDEFINED ? std::nullopt : std::optional<int>{v});
    }
    return result;
}

template <typename Derived>
group group_accessors<Derived>::include(std::span<int const> ranks) const {
    MPI_Group g   = MPI_GROUP_EMPTY;
    int       err = MPI_Group_incl(grp(), static_cast<int>(ranks.size()), ranks.data(), &g);
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return group::from_native(g);
}

template <typename Derived>
group group_accessors<Derived>::exclude(std::span<int const> ranks) const {
    MPI_Group g   = MPI_GROUP_EMPTY;
    int       err = MPI_Group_excl(grp(), static_cast<int>(ranks.size()), ranks.data(), &g);
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return group::from_native(g);
}

template <typename Derived>
group group_accessors<Derived>::include_ranges(std::span<std::array<int, 3>> ranges) const {
    MPI_Group g   = MPI_GROUP_EMPTY;
    // std::array<int,3> has the same layout as int[3].
    int err = MPI_Group_range_incl(
        grp(), static_cast<int>(ranges.size()), reinterpret_cast<int(*)[3]>(ranges.data()), &g
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return group::from_native(g);
}

template <typename Derived>
group group_accessors<Derived>::exclude_ranges(std::span<std::array<int, 3>> ranges) const {
    MPI_Group g   = MPI_GROUP_EMPTY;
    // std::array<int,3> has the same layout as int[3].
    int err = MPI_Group_range_excl(
        grp(), static_cast<int>(ranges.size()), reinterpret_cast<int(*)[3]>(ranges.data()), &g
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return group::from_native(g);
}

// ── Set algebra free functions ────────────────────────────────────────────────

/// @brief Create a new group containing processes in both `a` and `b`.
///
/// Calls `MPI_Group_intersection`.
/// @throws mpi_error on failure.
[[nodiscard]] inline group intersection(group_view a, group_view b) {
    MPI_Group g   = MPI_GROUP_EMPTY;
    int       err = MPI_Group_intersection(a.mpi_handle(), b.mpi_handle(), &g);
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return group::from_native(g);
}

/// @brief Create a new group containing processes in `a` that are not in `b`.
///
/// Calls `MPI_Group_difference`.
/// @throws mpi_error on failure.
[[nodiscard]] inline group difference(group_view a, group_view b) {
    MPI_Group g   = MPI_GROUP_EMPTY;
    int       err = MPI_Group_difference(a.mpi_handle(), b.mpi_handle(), &g);
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return group::from_native(g);
}

/// @brief Create a new group containing all processes in `a` or `b` (or both).
///
/// Calls `MPI_Group_union`.
/// @throws mpi_error on failure.
[[nodiscard]] inline group set_union(group_view a, group_view b) {
    MPI_Group g   = MPI_GROUP_EMPTY;
    int       err = MPI_Group_union(a.mpi_handle(), b.mpi_handle(), &g);
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return group::from_native(g);
}

} // namespace mpi::experimental
