// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <iterator>
#include <limits>
#include <ranges>

#include <kamping/types/reduce_ops.hpp>
#include <mpi.h>

#include "kamping/v2/collectives/allreduce.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/ops.hpp"
#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "kamping/v2/views/with_type_view.hpp"
#include "mpi/buffer.hpp"
#include "mpi/handle.hpp"

namespace dstl {

namespace detail {

/// @brief Stateless combine functor: picks the greater of two values under Comp/Proj.
///
/// Used to build a custom MPI_Op via make_op() for the generic (non-default Comp/Proj) case.
/// Comp and Proj are default-constructed on every invocation — matching make_op()'s contract
/// that only stateless functors survive the trip through MPI_Op_create.
///
/// Since kamping-types#808, ScopedFunctorOp (which make_op<T>() below builds on) combines by
/// copy-construction, not assignment, so T only needs to be copy-constructible here -- e.g.
/// std::pair<const K, V> (a std::map's/flat_hash_map's value type) works despite having a
/// deleted operator=. Deliberately not gated on std::is_trivially_copyable_v<T> either: even an
/// ordinary std::pair<int, int> is never trivially copyable (see
/// https://stackoverflow.com/q/58283694), so requiring it would reject the common case.
template <typename T, typename Comp, typename Proj>
struct max_combine {
    T operator()(T const& a, T const& b) const {
        return std::invoke(Comp{}, std::invoke(Proj{}, a), std::invoke(Proj{}, b)) ? b : a;
    }
};

/// @brief True when the default ordering (std::ranges::less / std::identity) applies AND T is
/// one of the MPI builtin arithmetic types, i.e. MPI_MAX can be used directly with no
/// MPI_Op_create overhead.
template <typename T, typename Comp, typename Proj>
concept has_builtin_max = std::same_as<Comp, std::ranges::less> && std::same_as<Proj, std::identity>
                          && kamping::types::mpi_operation_traits<kamping::ops::max<>, T>::is_builtin;

} // namespace detail

/// @brief Returns the largest element of the globally distributed range.
///
/// Distributed equivalent of std::ranges::max applied to the concatenation of all local
/// ranges across the communicator. The result is available on every rank.
///
/// Each rank computes its local maximum under comp/proj (or `sentinel` if its local range is
/// empty), then a single MPI_Allreduce combines the per-rank candidates. When comp/proj are
/// the defaults and T is an MPI builtin arithmetic type, this dispatches straight to MPI_MAX
/// (no MPI_Op_create involved); otherwise a custom MPI_Op is built from comp/proj via make_op,
/// which requires them to be stateless (default-constructible, as they already must be to have
/// default arguments here).
///
/// R must be a forward_range with a deducible MPI element type, and its value type T must be
/// copy-constructible (`std::copy_constructible<T>`). Notably T need not be assignable, so this
/// works directly on e.g. `std::pair<const K, V>` (the value type of iterating a std::map) --
/// see detail::max_combine's comment for why this is not additionally gated on
/// std::is_trivially_copyable_v<T>.
///
/// @param sentinel Value contributed on behalf of a rank whose local range is empty; must
///   compare not-greater than every real element under comp/proj. Defaults to
///   std::numeric_limits<T>::lowest() when std::numeric_limits<T> is specialized; for other T
///   an explicit sentinel must be supplied (see the overload below).
///
/// @pre sentinel must be a lower bound for every element under comp/proj.
/// @pre If every rank's range is empty (so every contribution is `sentinel`), the result is
///   `sentinel` rather than the undefined behavior std::ranges::max has on an empty range.
template <
    std::ranges::forward_range R,
    class Proj                                                                             = std::identity,
    std::indirect_strict_weak_order<std::projected<std::ranges::iterator_t<R>, Proj>> Comp = std::ranges::less,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>                            Comm = MPI_Comm>
    requires mpi::experimental::has_mpi_type<R> && std::copy_constructible<std::ranges::range_value_t<R>>
auto max(R&& r, Comp comp, Proj proj, Comm const& comm, std::ranges::range_value_t<R> sentinel)
    -> std::ranges::range_value_t<R> {
    using T = std::ranges::range_value_t<R>;

    kamping::v2::comm_view const cv{mpi::experimental::handle(comm)};

    T local_max = std::ranges::empty(r) ? sentinel : *std::ranges::max_element(r, std::ref(comp), std::ref(proj));

    if (cv.size() == 1) {
        return local_max;
    }

    MPI_Datatype const mpi_type = mpi::experimental::type(r);

    if constexpr (detail::has_builtin_max<T, Comp, Proj>) {
        kamping::v2::allreduce(
            kamping::v2::inplace,
            kamping::v2::views::ref_single(local_max) | kamping::v2::views::with_type(mpi_type),
            kamping::ops::max<>{},
            comm
        );
    } else {
        kamping::v2::allreduce(
            kamping::v2::inplace,
            kamping::v2::views::ref_single(local_max) | kamping::v2::views::with_type(mpi_type),
            kamping::v2::make_op<T>(detail::max_combine<T, Comp, Proj>{}, kamping::v2::commutative),
            comm
        );
    }

    return local_max;
}

/// @brief Overload defaulting comp/proj/comm/sentinel; requires std::numeric_limits<T> to be
/// specialized so std::numeric_limits<T>::lowest() is a meaningful sentinel. For T without a
/// numeric_limits specialization, call the overload above with an explicit sentinel.
template <
    std::ranges::forward_range R,
    class Proj                                                                             = std::identity,
    std::indirect_strict_weak_order<std::projected<std::ranges::iterator_t<R>, Proj>> Comp = std::ranges::less,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>                            Comm = MPI_Comm>
    requires mpi::experimental::has_mpi_type<R> && std::numeric_limits<std::ranges::range_value_t<R>>::is_specialized
             && std::copy_constructible<std::ranges::range_value_t<R>>
auto max(R&& r, Comp comp = {}, Proj proj = {}, Comm const& comm = MPI_COMM_WORLD) -> std::ranges::range_value_t<R> {
    return max(
        std::forward<R>(r),
        std::move(comp),
        std::move(proj),
        comm,
        std::numeric_limits<std::ranges::range_value_t<R>>::lowest()
    );
}

} // namespace dstl
