// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <utility>

namespace kamping::v2 {

/// The send and receive buffer of a blocking operation with both, e.g. `sendrecv` or `alltoallv`.
///
/// A buffer passed as an lvalue is borrowed: the member is a reference to it. A buffer passed as an rvalue was moved
/// into the operation and is handed back here by value. Unpack it with a structured binding:
///
/// @code
/// auto [_, recv] = kamping::v2::allgatherv(local, kamping::v2::auto_recv_v<int>(), comm);
/// @endcode
template <typename SBuf, typename RBuf>
struct result {
    SBuf send;
    RBuf recv;

    template <std::size_t I>
    decltype(auto) get() & {
        if constexpr (I == 0)
            return (send);
        else
            return (recv);
    }

    template <std::size_t I>
    decltype(auto) get() && {
        if constexpr (I == 0)
            return std::forward<SBuf>(send);
        else
            return std::forward<RBuf>(recv);
    }

    template <std::size_t I>
    decltype(auto) get() const& {
        if constexpr (I == 0)
            return (send);
        else
            return (recv);
    }
};

// Free get<I> overloads in kamping::v2 for ADL lookup during structured bindings.
template <std::size_t I, typename SBuf, typename RBuf>
decltype(auto) get(result<SBuf, RBuf>& r) {
    return r.template get<I>();
}

template <std::size_t I, typename SBuf, typename RBuf>
decltype(auto) get(result<SBuf, RBuf>&& r) {
    return std::move(r).template get<I>();
}

template <std::size_t I, typename SBuf, typename RBuf>
decltype(auto) get(result<SBuf, RBuf> const& r) {
    return r.template get<I>();
}

} // namespace kamping::v2

// tuple_size / tuple_element specializations required for structured bindings.
namespace std {

template <typename SBuf, typename RBuf>
struct tuple_size<kamping::v2::result<SBuf, RBuf>> : integral_constant<size_t, 2> {};

template <typename SBuf, typename RBuf>
struct tuple_element<0, kamping::v2::result<SBuf, RBuf>> {
    using type = SBuf;
};

template <typename SBuf, typename RBuf>
struct tuple_element<1, kamping::v2::result<SBuf, RBuf>> {
    using type = RBuf;
};

} // namespace std
