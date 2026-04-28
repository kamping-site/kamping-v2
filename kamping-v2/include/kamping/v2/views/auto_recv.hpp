#pragma once

#include <vector>

#include "kamping/v2/type_pool.hpp"
#include "kamping/v2/views/resize_view.hpp"

namespace kamping::v2 {

/// @brief Convenience factory for a fully-managed non-variadic receive buffer.
///
/// Returns an owned `Cont` wrapped in `views::resize`, which defers the actual
/// resize until the collective has inferred the receive count via `infer()`.
///
///   v2::recv(v2::auto_recv<int>(), 0, comm);
///
/// The default container is `std::vector<T>`. Pass a different container type as
/// the second template argument when needed.
/// @tparam T    Element type.
/// @tparam Cont Container type (must support `resize(n)` and `data()`).
template <typename T, typename Cont = std::vector<T>>
auto auto_recv() {
    return Cont{} | kamping::v2::views::resize;
}

/// @brief Convenience factory for a fully-managed non-variadic receive buffer with automatic type registration.
///
/// Like the zero-argument overload but additionally attaches the MPI datatype for `T`
/// by calling `pool.register_type<T>()`, making it suitable for non-builtin types.
///
///   v2::recv(v2::auto_recv<MyStruct>(pool), 0, comm);
///
/// @tparam T    Element type satisfying `has_static_type_v`.
/// @tparam Cont Container type (must support `resize(n)` and `data()`).
/// @param pool  Type pool used to commit and cache the MPI datatype for `T`.
template <typename T, typename Cont = std::vector<T>>
auto auto_recv(type_pool& pool) {
    return Cont{} | views::with_auto_pool(pool) | kamping::v2::views::resize;
}

} // namespace kamping::v2
