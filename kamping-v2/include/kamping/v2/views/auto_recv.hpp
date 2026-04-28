#pragma once

#include <vector>

#include "kamping/v2/type_pool.hpp"
#include "kamping/v2/views/resize_view.hpp"

namespace kamping::v2 {
template <typename T, typename Cont = std::vector<T>>
auto auto_recv() {
    return Cont{} | kamping::v2::views::resize;
}

template <typename T, typename Cont = std::vector<T>>
auto auto_recv(type_pool& pool) {
    return Cont{} | views::with_auto_pool(pool) | kamping::v2::views::resize;
}
} // namespace kamping::v2
