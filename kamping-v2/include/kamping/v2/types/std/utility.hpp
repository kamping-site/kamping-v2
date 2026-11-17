// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <utility>

#include "kamping/v2/types/as_struct.hpp"
#include "kamping/v2/types/datatype_traits.hpp"

/// @file
/// Describes `std::pair` by its members (`as_struct`), skipping the padding between them.
///
/// Opt-in: include it to give `std::pair`s a datatype. kamping/v2/types/std/unsafe/utility.hpp describes them as bytes
/// instead. Include at most one of the two, and the same one in every translation unit of a program; mixing them
/// violates the one-definition rule.

namespace kamping::v2 {

/// @brief A `std::pair` of types with a datatype is described by its members (`as_struct`).
template <typename First, typename Second>
    requires has_datatype<First> && has_datatype<Second>
struct datatype_traits<std::pair<First, Second>> : as_struct<std::pair<First, Second>> {};

} // namespace kamping::v2
