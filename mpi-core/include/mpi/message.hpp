// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstdio>
#include <exception>
#include <utility>

#include <mpi.h>

namespace mpi::experimental {

namespace detail {
/// Reports that a matched message was dropped without being received, then calls std::terminate().
///
/// A message matched by MPI_Mprobe/MPI_Improbe is removed from the matching queue and can only be received through
/// its handle; dropping the handle loses the message for good and typically deadlocks the sender or a later receive.
/// Like an active request, this is treated as a programming error (see terminate_on_active_request in request.hpp).
[[noreturn]] inline void terminate_on_unreceived_message() noexcept {
    std::fprintf(
        stderr,
        "kamping: mpi::experimental::message dropped while it still holds a matched message%s. Receive every matched "
        "message with mrecv() or imrecv() (or release() it) before it goes out of scope or is overwritten.\n",
        std::uncaught_exceptions() > 0 ? " (during exception unwinding)" : ""
    );
    std::fflush(stderr);
    std::terminate();
}
} // namespace detail

/// Owning, move-only wrapper around an MPI_Message.
///
/// Satisfies convertible_to_mpi_handle_ptr<MPI_Message>, so it can be filled by mprobe()/improbe() and consumed by
/// mrecv()/imrecv(), which reset it to MPI_MESSAGE_NULL:
///
///   message m;
///   mprobe(source, tag, comm, m);
///   mrecv(buf, m);
///
/// A matched message must be received explicitly, or handed off with release(); the library never receives or
/// discards one implicitly. Destroying or move-assigning over a message that still holds a matched message is a hard
/// error: it prints a diagnostic and calls std::terminate(), also during exception unwinding. MPI_MESSAGE_NO_PROC (the
/// result of probing MPI_PROC_NULL) holds no message and may be dropped freely.
class [[nodiscard]] message {
public:
    /// Constructs a null message (MPI_MESSAGE_NULL).
    message() noexcept = default;

    /// Takes over responsibility for receiving `msg`. The caller must not receive `msg` itself afterwards.
    explicit message(MPI_Message msg) noexcept : _message(msg) {}

    message(message const&)            = delete;
    message& operator=(message const&) = delete;

    message(message&& other) noexcept : _message(std::exchange(other._message, MPI_MESSAGE_NULL)) {}

    /// Takes over `other`'s message. Terminates if this still holds a matched message.
    message& operator=(message&& other) noexcept {
        if (is_matched()) {
            detail::terminate_on_unreceived_message();
        }
        _message = std::exchange(other._message, MPI_MESSAGE_NULL);
        return *this;
    }

    /// Terminates if this still holds a matched message (see class documentation).
    ~message() {
        if (is_matched()) {
            detail::terminate_on_unreceived_message();
        }
    }

    /// @return `true` if this holds MPI_MESSAGE_NULL (never matched, received, or released).
    [[nodiscard]] bool is_null() const noexcept {
        return _message == MPI_MESSAGE_NULL;
    }

    /// Gives up responsibility for the message and returns it; this becomes MPI_MESSAGE_NULL.
    [[nodiscard]] MPI_Message release() noexcept {
        return std::exchange(_message, MPI_MESSAGE_NULL);
    }

    /// @return The underlying `MPI_Message` handle (for `handle`).
    [[nodiscard]] MPI_Message mpi_handle() const noexcept {
        return _message;
    }

    /// @return Pointer to the underlying `MPI_Message` handle (for `handle_ptr`).
    [[nodiscard]] MPI_Message* mpi_handle_ptr() noexcept {
        return &_message;
    }

private:
    /// @return `true` if this holds a message that still has to be received.
    [[nodiscard]] bool is_matched() const noexcept {
        return _message != MPI_MESSAGE_NULL && _message != MPI_MESSAGE_NO_PROC;
    }

    MPI_Message _message = MPI_MESSAGE_NULL;
};

} // namespace mpi::experimental
