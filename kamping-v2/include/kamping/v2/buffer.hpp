// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

/// @file
/// The buffer contract in the `kamping::v2` namespace: accessors, buffer concepts and the concrete buffer types
/// `mpi_span` / `mpi_span_v`, re-exported from the core layer (`mpi/buffer.hpp`, `mpi/mpi_span.hpp`).
///
/// To adapt a type you don't own, specialize `mpi::experimental::buffer_traits` (a specialization must name the
/// namespace the template is declared in).
///
/// The mapping of builtin C++ types to predefined MPI datatypes is re-exported as well: the concept
/// `builtin_mpi_type<T>`, `builtin_datatype<T>()`, which returns the datatype (e.g. for the type constructors or raw
/// MPI calls), `representation_traits<T>`, which makes a class that wraps an arithmetic value a builtin type
/// (specialize it in `mpi::experimental`), and `unpacked_bool`, a `bool` for containers that MPI can send.

#pragma once

#include "mpi/buffer.hpp"
#include "mpi/builtin_types.hpp"
#include "mpi/mpi_span.hpp"

namespace kamping::v2 {
// Accessors
using mpi::experimental::count;
using mpi::experimental::counts;
using mpi::experimental::displs;
using mpi::experimental::ptr;
using mpi::experimental::type;

// Customization points
using mpi::experimental::buffer_traits;
using mpi::experimental::representation_traits;

// Builtin type mapping
using mpi::experimental::builtin_datatype;
using mpi::experimental::unpacked_bool;

// Concepts
using mpi::experimental::builtin_mpi_type;
using mpi::experimental::count_range;
using mpi::experimental::data_buffer;
using mpi::experimental::data_buffer_v;
using mpi::experimental::has_mpi_count;
using mpi::experimental::has_mpi_counts;
using mpi::experimental::has_mpi_counts_mutable;
using mpi::experimental::has_mpi_displs;
using mpi::experimental::has_mpi_ptr;
using mpi::experimental::has_mpi_type;
using mpi::experimental::recv_buffer;
using mpi::experimental::recv_buffer_v;
using mpi::experimental::send_buffer;
using mpi::experimental::send_buffer_v;
using mpi::experimental::send_recv_buffer;

// Concrete buffer types
using mpi::experimental::mpi_cspan;
using mpi::experimental::mpi_cspan_v;
using mpi::experimental::mpi_span;
using mpi::experimental::mpi_span_v;
} // namespace kamping::v2
