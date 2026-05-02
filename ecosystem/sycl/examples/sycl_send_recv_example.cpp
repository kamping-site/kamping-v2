#include <iostream>
#include <ranges>
#include <sycl/sycl.hpp>
#include <mpi/environment.hpp>

int main(int argc, char* argv[]) {
  mpi::environment env;
  sycl::queue q;
  std::cout << "Using " << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  sycl::buffer<int> buf{sycl::range{10}};

  static_assert(!std::ranges::range<decltype(buf)>);
  q.submit([&](sycl::handler& h) {
     sycl::accessor acc{buf, h};
     static_assert(std::ranges::sized_range<decltype(acc)>);
     static_assert(!std::ranges::contiguous_range<decltype(acc)>);
     h.parallel_for(acc.get_range(), [=](sycl::id<1> idx) { acc[idx] = 42; });
   }).wait();

  sycl::host_accessor acc(buf);
  static_assert(std::ranges::sized_range<decltype(acc)>);
  static_assert(!std::ranges::contiguous_range<decltype(acc)>);

  for (auto& x : acc) {
    std::cout << x << ",";
  }
  return 0;
}
