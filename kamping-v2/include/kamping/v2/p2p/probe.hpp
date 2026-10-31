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
#include "mpi/p2p/iprobe.hpp"
#include "mpi/p2p/mprobe.hpp"
#include "mpi/p2p/probe.hpp"

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

/// @brief Blocks until a message matching `source` and `tag` is available, without receiving it (`MPI_Probe`).
/// @param status Optional: filled with the status of the message. Defaults to `MPI_STATUS_IGNORE`.
template <
    mpi::experimental::rank                                      Source,
    mpi::experimental::tag                                       Tag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
void probe(Source source, Tag tag, Comm const& comm, Status&& status = MPI_STATUS_IGNORE) {
    mpi::experimental::probe(std::move(source), std::move(tag), comm, std::forward<Status>(status));
}

/// Like the overload above, probing for any tag.
template <
    mpi::experimental::rank                                      Source,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
void probe(Source source, Comm const& comm, Status&& status = MPI_STATUS_IGNORE) {
    mpi::experimental::probe(std::move(source), MPI_ANY_TAG, comm, std::forward<Status>(status));
}

/// Like the overload above, probing for any source and any tag.
template <
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
void probe(Comm const& comm, Status&& status = MPI_STATUS_IGNORE) {
    mpi::experimental::probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, std::forward<Status>(status));
}

/// @brief Checks whether a message matching `source` and `tag` is available, without blocking or receiving it
/// (`MPI_Iprobe`).
/// @param status Optional: filled with the status of the message if one is available. Defaults to `MPI_STATUS_IGNORE`.
/// @return `true` if a matching message is available.
template <
    mpi::experimental::rank                                      Source,
    mpi::experimental::tag                                       Tag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
[[nodiscard]] bool iprobe(Source source, Tag tag, Comm const& comm, Status&& status = MPI_STATUS_IGNORE) {
    return mpi::experimental::iprobe(std::move(source), std::move(tag), comm, std::forward<Status>(status));
}

/// Like the overload above, probing for any tag.
template <
    mpi::experimental::rank                                      Source,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
[[nodiscard]] bool iprobe(Source source, Comm const& comm, Status&& status = MPI_STATUS_IGNORE) {
    return mpi::experimental::iprobe(std::move(source), MPI_ANY_TAG, comm, std::forward<Status>(status));
}

/// Like the overload above, probing for any source and any tag.
template <
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
[[nodiscard]] bool iprobe(Comm const& comm, Status&& status = MPI_STATUS_IGNORE) {
    return mpi::experimental::iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, std::forward<Status>(status));
}

/// @brief Blocks until a message matching `source` and `tag` arrives and matches it (`MPI_Mprobe`).
///
/// The message is removed from the matching queue and must be received through the returned @ref probe_result.
/// @return The matched message and its status.
template <
    mpi::experimental::rank                                Source,
    mpi::experimental::tag                                 Tag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] probe_result mprobe(Source source, Tag tag, Comm const& comm) {
    probe_result res;
    mpi::experimental::mprobe(std::move(source), std::move(tag), comm, res, res.status());
    return res;
}

/// Like the overload above, matching a message with any tag.
template <mpi::experimental::rank Source, mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] probe_result mprobe(Source source, Comm const& comm) {
    return kamping::v2::mprobe(std::move(source), MPI_ANY_TAG, comm);
}

/// Like the overload above, matching a message from any source with any tag.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] probe_result mprobe(Comm const& comm) {
    return kamping::v2::mprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm);
}

/// @brief Matches a message matching `source` and `tag` if one is available, without blocking (`MPI_Improbe`).
///
/// A matched message is removed from the matching queue and must be received through the returned @ref probe_result.
/// @return The matched message and its status, or `std::nullopt` if no matching message is available.
template <
    mpi::experimental::rank                                Source,
    mpi::experimental::tag                                 Tag,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] std::optional<probe_result> improbe(Source source, Tag tag, Comm const& comm) {
    probe_result res;
    bool         probed = mpi::experimental::improbe(std::move(source), std::move(tag), comm, res, res.status());
    if (probed) {
        return res;
    }
    return std::nullopt;
}

/// Like the overload above, matching a message with any tag.
template <mpi::experimental::rank Source, mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] std::optional<probe_result> improbe(Source source, Comm const& comm) {
    return kamping::v2::improbe(std::move(source), MPI_ANY_TAG, comm);
}

/// Like the overload above, matching a message from any source with any tag.
template <mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm>
[[nodiscard]] std::optional<probe_result> improbe(Comm const& comm) {
    return kamping::v2::improbe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm);
}
} // namespace kamping::v2
