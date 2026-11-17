// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include "kamping/v2/types/datatype_traits.hpp"

/// @file
/// Describes `std::pair` as its bytes (`as_bytes`), including the padding between and after the members.
///
/// Opt-in and unsafe: sending the padding is usually faster than skipping it, but only correct if all processes use the
/// same representation of the pair. kamping/v2/types/std/utility.hpp describes pairs by their members instead. Include
/// at most one of the two, and the same one in every translation unit of a program; mixing them violates the
/// one-definition rule.

namespace kamping::v2 {

/// @brief A `std::pair` of types with a datatype is described as its bytes (`as_bytes`).
template <typename First, typename Second>
    requires has_datatype<First> && has_datatype<Second>
struct datatype_traits<std::pair<First, Second>> : as_bytes<std::pair<First, Second>> {};

} // namespace kamping::v2
