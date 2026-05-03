#include <iostream>
#include <ranges>
#include <sycl/sycl.hpp>

#include <kamping/ecosystem/sycl_view.hpp>
#include <kamping/v2/comm.hpp>
#include <kamping/v2/environment.hpp>
#include <kamping/v2/p2p/recv.hpp>
#include <kamping/v2/p2p/send.hpp>

// Demonstrates device-aware MPI with SYCL:
//   Rank 0 fills a buffer with 0..9 on device and sends it to rank 1.
//   Rank 1 squares each element on device and sends back.
//   Rank 0 reads the final result via a host accessor (no interop handle needed).
//
// Both paths of views::sycl are shown:
//   device accessor + interop handle  →  acc | views::sycl(ih)
//   host accessor                     →  hacc | views::sycl

int main(int argc, char* argv[]) {
    kamping::v2::environment env;
    kamping::v2::comm_view   world(MPI_COMM_WORLD);

    sycl::queue       q;
    sycl::buffer<int> buf{sycl::range{10}};
    std::cout << "[rank " << world.rank() << "] using "
              << q.get_device().get_info<sycl::info::device::name>() << '\n';

    if (world.rank() == 0) {
        // Fill with 0..9 on device.
        q.submit([&](sycl::handler& h) {
            sycl::accessor acc{buf, h, sycl::write_only};
            h.parallel_for(acc.get_range(), [=](sycl::id<1> idx) {
                acc[idx] = static_cast<int>(idx[0]);
            });
        });
        // Send to rank 1 via device-aware MPI.
        q.submit([&](sycl::handler& h) {
            sycl::accessor acc{buf, h, sycl::read_only};
            h.host_task([=](sycl::interop_handle ih) {
                kamping::v2::send(acc | kamping::v2::views::sycl(ih), 1, 0, world);
            });
        });
        // Receive squared values back from rank 1.
        q.submit([&](sycl::handler& h) {
            sycl::accessor acc{buf, h, sycl::write_only};
            h.host_task([=](sycl::interop_handle ih) {
                kamping::v2::recv(acc | kamping::v2::views::sycl(ih), world);
            });
        });
    } else if (world.rank() == 1) {
        // Receive from rank 0.
        q.submit([&](sycl::handler& h) {
            sycl::accessor acc{buf, h, sycl::write_only};
            h.host_task([=](sycl::interop_handle ih) {
                kamping::v2::recv(acc | kamping::v2::views::sycl(ih), world);
            });
        });
        // Square each element on device.
        q.submit([&](sycl::handler& h) {
            sycl::accessor acc{buf, h, sycl::read_write};
            h.parallel_for(acc.get_range(), [=](sycl::id<1> idx) {
                acc[idx] *= acc[idx];
            });
        });
        // Send squared values back via device-aware MPI.
        q.submit([&](sycl::handler& h) {
            sycl::accessor acc{buf, h, sycl::read_only};
            h.host_task([=](sycl::interop_handle ih) {
                kamping::v2::send(acc | kamping::v2::views::sycl(ih), 0, 0, world);
            });
        });
    }
    q.wait();

    // Rank 0 reads the final result via a host accessor (no interop handle needed).
    // sycl::host_accessor is sized but non-contiguous — views::sycl exposes it via
    // get_pointer() so MPI can access the underlying buffer directly.
    static_assert(std::ranges::sized_range<sycl::host_accessor<int, 1>>);
    static_assert(!std::ranges::contiguous_range<sycl::host_accessor<int, 1>>);

    if (world.rank() == 0) {
        sycl::host_accessor hacc{buf, sycl::read_only};
        std::cout << "squared: ";
        for (auto& x : hacc) {
            std::cout << x << ' ';
        }
        std::cout << '\n';  // expected: 0 1 4 9 16 25 36 49 64 81
    }
    return 0;
}
