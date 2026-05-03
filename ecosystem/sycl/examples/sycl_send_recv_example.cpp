#include <iostream>
#include <ranges>
#include <sycl/sycl.hpp>
#include <kamping/v2/environment.hpp>
#include <kamping/ecosystem/sycl_view.hpp>
#include <kamping/v2/comm.hpp>
#include <kamping/v2/p2p/send.hpp>
#include <kamping/v2/p2p/recv.hpp>

int main(int argc, char* argv[]) {
  kamping::v2::environment env;
  kamping::v2::comm_view world(MPI_COMM_WORLD);
  sycl::queue q;
  std::cout << "Using " << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  sycl::buffer<int> buf{sycl::range{10}};

  // static_assert(!std::ranges::range<decltype(buf)>);
  if (world.rank() == 0) {
    q.submit([&](sycl::handler& h) {
      sycl::accessor acc{buf, h, sycl::write_only};
      // static_assert(std::ranges::sized_range<decltype(acc)>);
      // static_assert(!std::ranges::contiguous_range<decltype(acc)>);
      h.parallel_for(acc.get_range(), [=](sycl::id<1> idx) { acc[idx] = 42; });
    });
    q.submit([&](sycl::handler& h) {
      sycl::accessor acc{buf, h, sycl::read_only};
      h.host_task([=](sycl::interop_handle ih) {
	kamping::v2::send(kamping::v2::sycl_view {acc, ih}, 1, 0, world);
      });
    });
    
  } else if (world.rank() == 1) {
    q.submit([&](sycl::handler& h) {
      sycl::accessor acc{buf, h, sycl::write_only};
      h.host_task([=](sycl::interop_handle ih) {
	kamping::v2::recv(kamping::v2::sycl_view {acc, ih}, world);
      });
    
    });
  }
  q.wait();
  

  sycl::host_accessor acc(buf);
  static_assert(std::ranges::sized_range<decltype(acc)>);
  static_assert(!std::ranges::contiguous_range<decltype(acc)>);

  for (auto& x : acc) {
    std::cout << x << ",";
  }
  return 0;
}
