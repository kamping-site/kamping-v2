// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <tuple>

#include "kamping/v2/types/datatype_traits.hpp"

/// @file
/// Describes `std::tuple` as its bytes (`as_bytes`), including the padding between and after the members.
///
/// Opt-in and unsafe: sending the padding is usually faster than skipping it, but only correct if all processes use the
/// same representation of the tuple. kamping/v2/types/std/tuple.hpp describes tuples by their members instead. Include
/// at most one of the two, and the same one in every translation unit of a program; mixing them violates the
/// one-definition rule.

namespace kamping::v2 {

/// @brief A non-empty `std::tuple` of types with a datatype is described as its bytes (`as_bytes`).
template <typename... Ts>
    requires(sizeof...(Ts) > 0) && (has_datatype<Ts> && ...)
struct datatype_traits<std::tuple<Ts...>> : as_bytes<std::tuple<Ts...>> {};

} // namespace kamping::v2
