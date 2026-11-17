// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <tuple>

#include "kamping/v2/types/as_struct.hpp"
#include "kamping/v2/types/datatype_traits.hpp"

/// @file
/// Describes `std::tuple` by its members (`as_struct`), skipping the padding between them.
///
/// Opt-in: include it to give `std::tuple`s a datatype. kamping/v2/types/std/unsafe/tuple.hpp describes them as bytes
/// instead. Include at most one of the two, and the same one in every translation unit of a program; mixing them
/// violates the one-definition rule.

namespace kamping::v2 {

/// @brief A non-empty `std::tuple` of types with a datatype is described by its members (`as_struct`).
template <typename... Ts>
    requires(sizeof...(Ts) > 0) && (has_datatype<Ts> && ...)
struct datatype_traits<std::tuple<Ts...>> : as_struct<std::tuple<Ts...>> {};

} // namespace kamping::v2
