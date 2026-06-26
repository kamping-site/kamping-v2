// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

/// @file
/// An allocator that default-initializes (rather than value-initializes) elements.
///
/// `std::vector<T>(n)` and `vector::resize(n)` value-initialize every element, which for a trivial
/// `T` means writing zeros to the whole buffer. For an intermediate buffer that is about to be
/// overwritten wholesale (an MPI recv target, or the destination of a `std::copy_n` rebin), that
/// initial zero-fill is pure overhead — and a serial one. Under MPI_THREAD_FUNNELED the surrounding
/// region is single-threaded, so the zero-fill cannot overlap with the OpenMP rebin that follows;
/// it shows up directly on the critical path. `default_init_allocator` skips it: trivial elements
/// are left uninitialized, non-trivial elements are still default-constructed (so correctness is
/// unchanged — only the redundant value-initialization of trivial types is elided).

namespace dstl::detail {

/// Allocator adaptor that replaces value-initialization with default-initialization. See @file.
template <typename T, typename Alloc = std::allocator<T>>
class default_init_allocator : public Alloc {
    using a_t = std::allocator_traits<Alloc>;

public:
    /// Rebind to the same adaptor over `U`, wrapping the rebound underlying allocator.
    template <typename U>
    struct rebind {
        using other = default_init_allocator<U, typename a_t::template rebind_alloc<U>>;
    };

    // Inherit the underlying allocator's constructors.
    using Alloc::Alloc;

    /// Default-initialize `*ptr` (no zero-fill for trivial `U`).
    template <typename U>
    void construct(U* ptr) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(ptr)) U;
    }

    /// Forward to the underlying allocator for any value-/argument-construction.
    template <typename U, typename... Args>
    void construct(U* ptr, Args&&... args) {
        a_t::construct(static_cast<Alloc&>(*this), ptr, std::forward<Args>(args)...);
    }
};

/// A `std::vector` whose elements are default- rather than value-initialized. Use for intermediate
/// buffers that are fully overwritten before being read.
template <typename T>
using uninit_vector = std::vector<T, default_init_allocator<T>>;

} // namespace dstl::detail
