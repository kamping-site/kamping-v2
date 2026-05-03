/// CUDA-aware MPI send/recv with thrust::device_vector via kamping.
///
/// Demonstrates two ways to use thrust::device_vector with kamping:
///   1. Include thrust_buffer_traits.hpp — device_vector works directly, no
///   piping needed.
///   2. Pipe through views::thrust — explicit per-call opt-in.
///
/// Requires an MPI implementation built with CUDA support (OpenMPI with
/// `--with-cuda`, MPICH with GPU-aware build, etc.). The MPI pointer passed in
/// is a raw device pointer obtained from `thrust::device_vector::data().get()`.

#include <cstddef>
#include <iostream>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

// Option A: include buffer traits — device_vector usable directly everywhere.
#include "kamping/ecosystem/thrust_buffer_traits.hpp"
// Option B: include the view adaptor — pipe explicitly per call.
#include "kamping/ecosystem/thrust_device_view.hpp"
#include "kamping/v2/environment.hpp"
#include "kamping/v2/p2p/recv.hpp"
#include "kamping/v2/p2p/send.hpp"
#include "kamping/v2/views/resize_view.hpp"
#include "mpi/comm.hpp"

// Intel MPI does not support GPU-aware matched receives (MPI_Mrecv on device
// memory). Opt out so kamping falls back to plain MPI_Probe + MPI_Recv for
// deferred recvs.
template <typename T, typename Alloc>
inline constexpr bool
    kamping::v2::supports_matched_probe<thrust::device_vector<T, Alloc>> =
        false;

template <typename HostVec>
void print_vec(HostVec const& v) {
  std::cout << "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::cout << v[i] << ((i + 1 < v.size()) ? ", " : "");
  }
  std::cout << "]\n";
}

int main(int argc, char* argv[]) {
  kamping::v2::environment env(argc, argv);
  mpi::experimental::comm_view const world{MPI_COMM_WORLD};

  KAMPING_ASSERT(world.size() == 2uz,
                 "This example must be run with exactly 2 ranks.");

  constexpr std::size_t N = 8;

  // Example 1 (buffer_traits): device_vector passed directly — no piping.
  if (world.rank() == 0) {
    thrust::device_vector<int> d_send(N);
    thrust::sequence(d_send.begin(), d_send.end(), 100);
    kamping::v2::send(d_send, 1, 0, world);
  } else {
    thrust::device_vector<int> d_recv(N);
    kamping::v2::recv(d_recv, 0, 0, world);

    thrust::host_vector<int> h(d_recv.size());
    thrust::copy(d_recv.begin(), d_recv.end(), h.begin());
    print_vec(h);
  }

  // Example 2 (buffer_traits): deferred-size receive — pipe through
  // views::resize.
  if (world.rank() == 0) {
    thrust::device_vector<int> d_send(N);
    thrust::sequence(d_send.begin(), d_send.end(), 200);
    kamping::v2::send(d_send, 1, 0, world);
  } else {
    auto d_recv = kamping::v2::recv(
        thrust::device_vector<int>{} | kamping::v2::views::resize, 0, 0, world);

    thrust::host_vector<int> h(d_recv.size());
    thrust::copy(d_recv.begin(), d_recv.end(), h.begin());
    print_vec(h);
  }

  // Example 3 (view adaptor): explicit per-call opt-in via views::thrust.
  if (world.rank() == 0) {
    thrust::device_vector<int> d_send(N);
    thrust::sequence(d_send.begin(), d_send.end(), 300);
    kamping::v2::send(d_send | kamping::v2::views::thrust, 1, 0, world);
  } else {
    thrust::device_vector<int> d_recv(N);
    kamping::v2::recv(d_recv | kamping::v2::views::thrust, 0, 0, world);

    thrust::host_vector<int> h(d_recv.size());
    thrust::copy(d_recv.begin(), d_recv.end(), h.begin());
    print_vec(h);
  }

  return 0;
}
