#include <iostream>
#include <ranges>
#include <sycl/sycl.hpp>

#include <kamping/ecosystem/sycl_view.hpp>
#include <kamping/v2/comm.hpp>
#include <kamping/v2/environment.hpp>
#include <kamping/v2/p2p/recv.hpp>
#include <kamping/v2/p2p/send.hpp>

// Demonstrates both paths of views::sycl:
//
//   device accessor + interop handle  →  acc | views::sycl(ih)
//     Rank 0 fills 0..9 on device and sends to rank 1 via device-aware MPI.
//     Rank 1 receives into device memory and squares each element on device.
//
//   host accessor (no handle needed)  →  hacc | views::sycl
//     Rank 1 sends the result back to rank 0 via a host accessor.
//     Rank 0 receives into a host accessor and prints the result.
//
// Expected output: squared: 0 1 4 9 16 25 36 49 64 81

// sycl::host_accessor is sized but non-contiguous — views::sycl exposes it via
// get_pointer() rather than a range-based data() fallback.
static_assert(std::ranges::sized_range<sycl::host_accessor<int, 1>>);
static_assert(!std::ranges::contiguous_range<sycl::host_accessor<int, 1>>);

int main(int argc, char* argv[]) {
    kamping::v2::environment env;
    kamping::v2::comm_view   world(MPI_COMM_WORLD);

    sycl::queue       q;
    sycl::buffer<int> buf{sycl::range{10}};
    std::cout << "[rank " << world.rank() << "] using "
              << q.get_device().get_info<sycl::info::device::name>() << '\n';

    if (world.rank() == 0) {
        // Fill with 0..9 on device and send via device-aware MPI.
        q.submit([&](sycl::handler& h) {
            sycl::accessor acc{buf, h, sycl::write_only};
            h.parallel_for(acc.get_range(), [=](sycl::id<1> idx) {
                acc[idx] = static_cast<int>(idx[0]);
            });
        });
        q.submit([&](sycl::handler& h) {
            sycl::accessor acc{buf, h, sycl::read_only};
            h.host_task([=](sycl::interop_handle ih) {
                kamping::v2::send(acc | kamping::v2::views::sycl(ih), 1, 0, world);
            });
        });
        q.wait();

        // Receive the squared result from rank 1 via host accessor.
        sycl::host_accessor hacc{buf, sycl::write_only};
        kamping::v2::recv(hacc | kamping::v2::views::sycl, world);
        std::cout << "squared: ";
        for (auto& x : hacc) {
            std::cout << x << ' ';
        }
        std::cout << '\n';
    } else if (world.rank() == 1) {
        // Receive from rank 0 into device memory and square each element.
        q.submit([&](sycl::handler& h) {
            sycl::accessor acc{buf, h, sycl::write_only};
            h.host_task([=](sycl::interop_handle ih) {
                kamping::v2::recv(acc | kamping::v2::views::sycl(ih), world);
            });
        });
        q.submit([&](sycl::handler& h) {
            sycl::accessor acc{buf, h, sycl::read_write};
            h.parallel_for(acc.get_range(), [=](sycl::id<1> idx) {
                acc[idx] *= acc[idx];
            });
        });
        q.wait();

        // Send result back to rank 0 via host accessor.
        sycl::host_accessor hacc{buf, sycl::read_only};
        kamping::v2::send(hacc | kamping::v2::views::sycl, 0, 0, world);
    }
    return 0;
}
