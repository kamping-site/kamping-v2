#pragma once

#include <memory>

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

/// Non-owning wrapper around an MPI_Request*.
///
/// Satisfies convertible_to_mpi_handle_ptr<MPI_Request> so it can be passed to
/// any mpi::experimental:: non-blocking operation as the request output parameter.
/// Does not complete or free the request on destruction — use iresult for RAII.
///
/// wait() and test() follow the same semantics as iresult: wait() blocks until
/// the operation completes; test() polls and returns true when done. Both accept
/// an optional status out-parameter via convertible_to_mpi_handle_ptr<MPI_Status>
/// (raw MPI_Status*, status_view, or omitted to use MPI_STATUS_IGNORE).
/// Since request_view is non-owning, wait() returns void — the buffer is the
/// caller's responsibility.
class request_view {
public:
    explicit request_view(MPI_Request& req) noexcept : _req(std::addressof(req)) {}

    [[nodiscard]] MPI_Request* mpi_handle_ptr() noexcept { return _req; }

    template <convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    void wait(Status&& status = MPI_STATUS_IGNORE) {
        int err = MPI_Wait(_req, handle_ptr(status));
        if (err != MPI_SUCCESS) throw mpi_error(err);
    }

    template <convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    bool test(Status&& status = MPI_STATUS_IGNORE) {
        int flag;
        int err = MPI_Test(_req, &flag, handle_ptr(status));
        if (err != MPI_SUCCESS) throw mpi_error(err);
        return static_cast<bool>(flag);
    }

private:
    MPI_Request* _req;
};

static_assert(convertible_to_mpi_handle_ptr<request_view, MPI_Request>);

} // namespace mpi::experimental
