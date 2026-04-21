#pragma once

#include <vector>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/auto_counts_view.hpp"
#include "kamping/v2/views/auto_displs_view.hpp"
#include "kamping/v2/views/resize_v_view.hpp"

namespace kamping::v2::views {

/// Pipe adaptor combining auto_counts() | auto_displs() | resize_v in a single step.
///
/// Attaches owned counts and displacement buffers to the base data buffer and marks it
/// for automatic resize on receive. The full deferred protocol (set_comm_size / MPI writes
/// into counts / commit_counts / lazy displ computation / resize on mpi_ptr) is provided
/// by the composed views.
///
/// Usage:
///   recv_buf | views::auto_recv_v                      // lvalue — borrows recv_buf
///   v2::auto_recv_v<int>()                             // owned factory form
inline constexpr struct auto_recv_v_fn : kamping::v2::adaptor_closure<auto_recv_v_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return std::forward<R>(r) | kamping::v2::views::auto_counts() | kamping::v2::views::auto_displs()
               | kamping::v2::views::resize_v;
    }
} auto_recv_v{};

} // namespace kamping::v2::views

namespace kamping::v2 {

/// Convenience factory for a fully-managed variadic receive buffer.
///
/// Returns an owned Cont wrapped in views::auto_recv_v, which provides automatic
/// per-rank count exchange, displacement computation, and buffer resizing.
///
///   v2::allgatherv(send_data, v2::auto_recv_v<int>(), comm);
///   v2::alltoallv(sbuf,       v2::auto_recv_v<float>(), comm);
///
/// The default container is std::vector<T>. Substitute a different container type as
/// the second argument when needed.
template <typename T, typename Cont = std::vector<T>>
auto auto_recv_v() {
    return Cont{} | kamping::v2::views::auto_recv_v;
}

} // namespace kamping::v2
