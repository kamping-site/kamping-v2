// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <type_traits>
#include <utility>

#include <mpi.h>

#include "kamping/v2/infer.hpp"
#include "kamping/v2/iresult.hpp"
#include "kamping/v2/p2p/probe.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "mpi/handle.hpp"
#include "mpi/p2p/imrecv.hpp"
#include "mpi/p2p/mrecv.hpp"

// mpi::experimental::mrecv / imrecv are deliberately not brought into kamping::v2: they accept anything convertible to
// an MPI_Message*, including an lvalue probe_result, and would then receive without sizing a deferred recv buffer.

namespace kamping::v2 {

/// @brief Receives the message matched by `probe` into `rbuf` (`MPI_Mrecv`).
///
/// Calls the `infer(comm_op::mrecv, ...)` customization point first; the default overload sizes a deferred recv
/// buffer (e.g. `views::resize`) from the probed count, so no further probing is needed. Takes `probe` by value, so
/// the caller must hand it over with `std::move` — the message can be received only once.
/// @param status Optional: filled with the status of the receive. Defaults to `MPI_STATUS_IGNORE`.
/// @return The buffer (ownership semantics match blocking recv).
template <
    mpi::experimental::recv_buffer                               RBuf,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto mrecv(RBuf&& rbuf, probe_result probe, Status&& status = MPI_STATUS_IGNORE) -> RBuf {
    static_assert(
        kamping::v2::supports_matched_probe<std::remove_cvref_t<RBuf>>,
        "This buffer type does not support matched receives (supports_matched_probe is false)."
    );
    infer(comm_op::mrecv{}, rbuf, std::as_const(probe));
    mpi::experimental::mrecv(rbuf, probe, std::forward<Status>(status));
    return std::forward<RBuf>(rbuf);
}

/// @brief Starts a non-blocking receive of the message matched by `probe` into `rbuf`, using a caller-supplied request
/// (`MPI_Imrecv`).
///
/// Like @ref mrecv, sizes a deferred recv buffer from the probed count before the receive is started. The message is
/// consumed immediately, so `probe` need not outlive the request; the caller must keep `rbuf` alive and complete
/// `request`.
/// @return The buffer (ownership semantics match blocking recv).
template <mpi::experimental::recv_buffer RBuf, mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request>
auto imrecv(Request&& request, RBuf&& rbuf, probe_result probe) -> RBuf {
    static_assert(
        kamping::v2::supports_matched_probe<std::remove_cvref_t<RBuf>>,
        "This buffer type does not support matched receives (supports_matched_probe is false)."
    );
    infer(comm_op::mrecv{}, rbuf, std::as_const(probe));
    mpi::experimental::imrecv(rbuf, probe, std::forward<Request>(request));
    return std::forward<RBuf>(rbuf);
}

/// @brief Starts a non-blocking receive of the message matched by `probe` into `rbuf` (`MPI_Imrecv`).
///
/// Like @ref mrecv, sizes a deferred recv buffer from the probed count before the receive is started. The message is
/// consumed immediately, so `probe` need not outlive the returned iresult, which owns the request and keeps the buffer
/// alive until completion.
/// @return An iresult to wait on or test for the received buffer.
template <mpi::experimental::recv_buffer RBuf>
auto imrecv(RBuf&& rbuf, probe_result probe) -> iresult<RBuf> {
    static_assert(
        kamping::v2::supports_matched_probe<std::remove_cvref_t<RBuf>>,
        "This buffer type does not support matched receives (supports_matched_probe is false)."
    );
    iresult<RBuf> res{std::forward<RBuf>(rbuf)};
    infer(comm_op::mrecv{}, res.view(), std::as_const(probe));
    mpi::experimental::imrecv(res.view(), probe, res.mpi_handle_ptr());
    return res;
}

} // namespace kamping::v2
