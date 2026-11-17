// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

/// @file
/// The native handle bridge in the `kamping::v2` namespace, re-exported from the core layer (`mpi/handle.hpp`):
/// extracting raw MPI handles, ranks and tags from wrapper types. Also re-exports the typed special values
/// (`any_source`, `any_tag`, `proc_null`, `undefined`) and the role concepts from `mpi/special_values.hpp`.
///
/// To adapt a type you don't own, specialize `mpi::experimental::handle_traits`, `handle_range_traits`,
/// `rank_traits` or `tag_traits` (a specialization must name the namespace the template is declared in).

#pragma once

#include "mpi/handle.hpp"
#include "mpi/special_values.hpp"

namespace kamping::v2 {
// Accessors
using mpi::experimental::handle;
using mpi::experimental::handle_ptr;
using mpi::experimental::handle_range;
using mpi::experimental::to_rank;
using mpi::experimental::to_tag;

// Customization points
using mpi::experimental::handle_range_traits;
using mpi::experimental::handle_traits;
using mpi::experimental::rank_traits;
using mpi::experimental::tag_traits;

// Concepts
using mpi::experimental::builtin_handle;
using mpi::experimental::convertible_to_mpi_handle;
using mpi::experimental::convertible_to_mpi_handle_ptr;
using mpi::experimental::convertible_to_mpi_handle_range;
using mpi::experimental::destination;
using mpi::experimental::message_tag;
using mpi::experimental::rank;
using mpi::experimental::sender;
using mpi::experimental::tag;

// Special values
using mpi::experimental::any_source;
using mpi::experimental::any_source_t;
using mpi::experimental::any_tag;
using mpi::experimental::any_tag_t;
using mpi::experimental::proc_null;
using mpi::experimental::proc_null_t;
using mpi::experimental::undefined;
using mpi::experimental::undefined_t;
} // namespace kamping::v2
