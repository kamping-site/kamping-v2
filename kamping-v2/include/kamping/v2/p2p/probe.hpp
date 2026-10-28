// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <optional>
#include <utility>

#include <mpi.h>

#include "kamping/v2/status.hpp"
#include "mpi/handle.hpp"
#include "mpi/message.hpp"
#include "mpi/p2p/improbe.hpp"
#include "mpi/p2p/mprobe.hpp"

namespace kamping::v2 {
/// @brief Result of a matched probe: the matched message together with its status.
///
/// Provides the status accessors (`source()`, `tag()`, `count()`) directly and satisfies
/// `convertible_to_mpi_handle_ptr<MPI_Message>`, so it can be passed to `mrecv()` / `imrecv()` to receive the message.
/// Move-only: like `mpi::experimental::message`, it terminates if destroyed while the matched message has not been
/// received or released.
class probe_result : public status_accessors<probe_result> {
public:
    /// @return A read-only view of the status of the matched message.
    const_status_view status() const {
        return _status;
    }

    /// @return A view of the status of the matched message.
    status_view status() {
        return _status;
    }

    /// @return The matched `MPI_Message` handle (for `handle`).
    [[nodiscard]] MPI_Message mpi_handle() const {
        return _message.mpi_handle();
    }

    /// @return Pointer to the matched `MPI_Message` handle (for `handle_ptr`).
    [[nodiscard]] MPI_Message* mpi_handle_ptr() {
        return _message.mpi_handle_ptr();
    }

    /// Gives up responsibility for the matched message and returns it; see mpi::experimental::message::release().
    [[nodiscard]] MPI_Message release() noexcept {
        return _message.release();
    }

private:
    friend class status_accessors<probe_result>;
    [[nodiscard]] MPI_Status const* status_ptr() const {
        return &_status.native();
    }
    mpi::experimental::message _message;
    kamping::v2::status        _status;
};

using mpi::experimental::mprobe;

template <
    mpi::experimental::rank                                Source = int,
    mpi::experimental::tag                                 Tag    = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm   = MPI_Comm>
/// @brief Blocks until a message matching `source` and `tag` arrives and matches it (`MPI_Mprobe`).
///
/// The message is removed from the matching queue and must be received through the returned @ref probe_result.
/// @return The matched message and its status.
[[nodiscard]] probe_result
mprobe(Source source = MPI_ANY_SOURCE, Tag tag = MPI_ANY_TAG, Comm const& comm = MPI_COMM_WORLD) {
    probe_result res;
    mpi::experimental::mprobe(std::move(source), std::move(tag), comm, res, res.status());
    return res;
}

using mpi::experimental::improbe;

template <
    mpi::experimental::rank                                Source = int,
    mpi::experimental::tag                                 Tag    = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm   = MPI_Comm>
/// @brief Matches a message matching `source` and `tag` if one is available, without blocking (`MPI_Improbe`).
///
/// A matched message is removed from the matching queue and must be received through the returned @ref probe_result.
/// @return The matched message and its status, or `std::nullopt` if no matching message is available.
[[nodiscard]] std::optional<probe_result>
improbe(Source source = MPI_ANY_SOURCE, Tag tag = MPI_ANY_TAG, Comm const& comm = MPI_COMM_WORLD) {
    probe_result res;
    bool         probed = mpi::experimental::improbe(std::move(source), std::move(tag), comm, res, res.status());
    if (probed) {
        return res;
    }
    return std::nullopt;
}
} // namespace kamping::v2
