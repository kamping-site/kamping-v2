// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

/// @file
/// Operation tags for the infer() customization point (see infer.hpp).
///
/// Each `kamping::v2::` wrapper calls `infer(comm_op::<operation>{}, ...)` before issuing its MPI call. Tags are types
/// rather than enumerators so that overloads can be added for new operations without modifying this header.

#pragma once

namespace kamping::v2::comm_op {

struct recv {};       ///< recv, irecv
struct mrecv {};      ///< mrecv, imrecv
struct bcast {};      ///< bcast
struct allgather {};  ///< allgather
struct allgatherv {}; ///< allgatherv
struct alltoall {};   ///< alltoall
struct alltoallv {};  ///< alltoallv
struct sendrecv {};   ///< sendrecv, isendrecv
struct reduce {};     ///< reduce
struct allreduce {};  ///< allreduce
struct gather {};     ///< gather
struct gatherv {};    ///< gatherv
struct scatter {};    ///< scatter
struct scatterv {};   ///< scatterv
struct scan {};       ///< scan
struct exscan {};     ///< exscan

} // namespace kamping::v2::comm_op
