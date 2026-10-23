// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstdio>
#include <exception>
#include <memory>
#include <utility>

#include <mpi.h>

#include "mpi/completion/wait.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

/// Non-owning wrapper around an MPI_Request*.
///
/// Satisfies convertible_to_mpi_handle_ptr<MPI_Request> so it can be passed to
/// any mpi::experimental:: non-blocking operation as the request output parameter.
/// Does not complete or free the request on destruction — use request or iresult for RAII.
///
/// wait() and test() are shorthands for the free functions in mpi/completion/wait.hpp and
/// follow the same semantics as iresult: wait() blocks until
/// the operation completes; test() polls and returns true when done. Both accept
/// an optional status out-parameter via convertible_to_mpi_handle_ptr<MPI_Status>
/// (raw MPI_Status*, status_view, or omitted to use MPI_STATUS_IGNORE).
/// Since request_view is non-owning, wait() returns void — the buffer is the
/// caller's responsibility.
class request_view {
public:
    explicit request_view(MPI_Request& req) noexcept : _req(std::addressof(req)) {}

    [[nodiscard]] MPI_Request* mpi_handle_ptr() noexcept {
        return _req;
    }

    template <convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    void wait(Status&& status = MPI_STATUS_IGNORE) {
        mpi::experimental::wait(_req, std::forward<Status>(status));
    }

    template <convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    bool test(Status&& status = MPI_STATUS_IGNORE) {
        return mpi::experimental::test(_req, std::forward<Status>(status));
    }

private:
    MPI_Request* _req;
};

static_assert(convertible_to_mpi_handle_ptr<request_view, MPI_Request>);

namespace detail {
/// Reports that an owning request type (`what`) dropped a request that is still active, then calls std::terminate().
/// Shared by request, request_vector and (through request) iresult.
///
/// Silently completing the request would block (and possibly deadlock), and silently dropping it would let MPI keep
/// using a buffer that is about to be released. Both hide a missing wait()/test(), so this is a hard error.
///
/// std::terminate() rather than MPI_Abort: like a joinable std::thread being destroyed, this is a programming error,
/// and terminate works in any MPI state (Sessions, after MPI_Finalize) without needing a communicator. Applications
/// that want all ranks killed reliably can install a handler via std::set_terminate that calls MPI_Abort.
[[noreturn]] inline void terminate_on_active_request(char const* what) noexcept {
    std::fprintf(
        stderr,
        "kamping: %s dropped while a request is still active%s. Complete every request with wait() or test() (or "
        "release() it) before it goes out of scope or is overwritten.\n",
        what,
        std::uncaught_exceptions() > 0 ? " (during exception unwinding)" : ""
    );
    std::fflush(stderr);
    std::terminate();
}
} // namespace detail

/// Owning, move-only wrapper around an MPI_Request.
///
/// Satisfies convertible_to_mpi_handle_ptr<MPI_Request>, so it can be passed directly to any
/// mpi::experimental:: non-blocking operation as the request output parameter:
///
///   request r;
///   isend(buf, dest, tag, comm, r);
///   r.wait();
///
/// The request must be completed explicitly — with wait(), a successful test(), or the free completion functions —
/// or handed off with release(); the library never completes a request implicitly. Destroying or move-assigning over
/// a request that is still active is a hard error: it prints a diagnostic and calls std::terminate(), also during
/// exception unwinding. Unlike iresult,
/// request does not own the buffer: the caller must keep it alive until the request completes.
///
/// Ownership can be handed back to raw MPI code with release(), and moved into or out of a request_vector with
/// request_vector::push_back(request&&) / request_vector::extract().
class [[nodiscard]] request {
public:
    /// Constructs a null request (MPI_REQUEST_NULL).
    request() noexcept = default;

    /// Takes over responsibility for completing `req`. The caller must not complete `req` itself afterwards.
    explicit request(MPI_Request req) noexcept : _req(req) {}

    request(request const&)            = delete;
    request& operator=(request const&) = delete;

    request(request&& other) noexcept : _req(std::exchange(other._req, MPI_REQUEST_NULL)) {}

    /// Takes over `other`'s request. Terminates if this still holds an active request.
    request& operator=(request&& other) noexcept {
        if (_req != MPI_REQUEST_NULL) {
            detail::terminate_on_active_request("mpi::experimental::request");
        }
        _req = std::exchange(other._req, MPI_REQUEST_NULL);
        return *this;
    }

    /// Terminates if the request is still active (see class documentation).
    ~request() {
        if (_req != MPI_REQUEST_NULL) {
            detail::terminate_on_active_request("mpi::experimental::request");
        }
    }

    /// @return `true` if this holds MPI_REQUEST_NULL (never started, completed, or released).
    [[nodiscard]] bool is_null() const noexcept {
        return _req == MPI_REQUEST_NULL;
    }

    /// Gives up responsibility for the request and returns it; this becomes MPI_REQUEST_NULL.
    [[nodiscard]] MPI_Request release() noexcept {
        return std::exchange(_req, MPI_REQUEST_NULL);
    }

    [[nodiscard]] MPI_Request* mpi_handle_ptr() noexcept {
        return &_req;
    }

    /// @return A non-owning view of this request.
    operator request_view() noexcept {
        return request_view{_req};
    }

    /// Shorthand for mpi::experimental::wait(*this, status).
    template <convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    void wait(Status&& status = MPI_STATUS_IGNORE) {
        mpi::experimental::wait(_req, std::forward<Status>(status));
    }

    /// Shorthand for mpi::experimental::test(*this, status).
    template <convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    bool test(Status&& status = MPI_STATUS_IGNORE) {
        return mpi::experimental::test(_req, std::forward<Status>(status));
    }

private:
    MPI_Request _req = MPI_REQUEST_NULL;
};

static_assert(convertible_to_mpi_handle_ptr<request, MPI_Request>);

} // namespace mpi::experimental
