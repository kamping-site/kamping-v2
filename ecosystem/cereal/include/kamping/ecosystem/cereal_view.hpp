// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>

#include <cereal/archives/binary.hpp>

#include "kamping/types/builtin_types.hpp"
#include "kamping/v2/views/adaptor.hpp"

namespace kamping::v2 {

/// Wraps an object and serializes/deserializes it with cereal for MPI transport.
///
/// @tparam T       The wrapped type: a (possibly const) lvalue reference for non-owning
///                 views, or a value type for owning views.
/// @tparam Alloc   Allocator for the internal byte buffer (default: `std::allocator<char>`).
///
/// Construct via `obj | kamping::views::serialize` (non-owning, lvalue) or
/// `kamping::v2::views::deserialize<T>()` (owning, default-constructed recv target).
///
/// **Send path** — `mpi_count()` / `mpi_ptr() const` lazily serialize the wrapped object
/// into `buffer_` on first access; the `ostringstream` result is moved (no copy) into
/// `buffer_`. A const view satisfies `send_buffer` but not `recv_buffer`.
///
/// **Recv path** — `set_recv_count(n)` records the incoming byte count; `mpi_ptr()`
/// (non-const) lazily resizes `buffer_` to that size so MPI can write into it directly;
/// `operator*` / `unwrap()` then trigger lazy deserialization via a zero-copy `membuf`
/// streambuf and clear `buffer_` to leave a deterministic empty state. Requires
/// non-const `T`; a non-const view satisfies both `send_buffer` and `recv_buffer`.
///
/// Range semantics are intentionally omitted. Access the wrapped object via `operator*` or
/// `operator->`; for ranges, dereference first: `for (auto& x : *view) { … }`.
template <typename T, typename Alloc = std::allocator<char>>
class serialization_view {
    static constexpr bool is_owning = !std::is_lvalue_reference_v<T>;
    using value_type                = std::remove_reference_t<T>;
    // Lvalue-ref case: store a (possibly const) pointer to avoid requiring copyability.
    // Owning case: store by value.
    using stored_t = std::conditional_t<is_owning, value_type, value_type*>;

    mutable stored_t base_;

    mutable std::basic_string<char, std::char_traits<char>, Alloc> buffer_;
    mutable bool                                                   serialized_            = false;
    mutable bool                                                   needs_deserialization_ = false;
    bool                                                           needs_resize_          = false;
    std::ptrdiff_t                                                 recv_count_            = 0;

    // base_ is mutable, so this is safe from const methods.
    // The const-lvalue-ref case (value_type is const-qualified) is prevented from
    // reaching do_deserialize() by the set_recv_count requires-clause.
    value_type& base_ref() const noexcept {
        if constexpr (is_owning)
            return base_;
        else
            return *base_;
    }

    void do_serialize() const {
        std::basic_ostringstream<char, std::char_traits<char>, Alloc> oss;
        {
            cereal::BinaryOutputArchive ar(oss);
            ar(base_ref());
        }
        buffer_     = std::move(oss).str(); // move when allocators compare equal
        serialized_ = true;
    }

    // Zero-copy read-only streambuf over an existing char buffer.
    // std::basic_ispanstream (C++23) would be cleaner but isn't universally available yet.
    struct membuf : std::basic_streambuf<char> {
        membuf(char const* data, std::size_t size) {
            auto p = const_cast<char*>(data); // setg requires non-const; we only read
            setg(p, p, p + size);
        }
    };

    void do_deserialize() const {
        membuf                     mb(buffer_.data(), buffer_.size());
        std::basic_istream<char>   is(&mb);
        cereal::BinaryInputArchive ar(is);
        ar(base_ref());
        buffer_.clear(); // received bytes no longer needed; known empty state
        needs_deserialization_ = false;
    }

public:
    /// Non-owning constructor: stores a pointer to the referenced object.
    /// Handles both `T&` and `T const&` (value_type may be const-qualified).
    explicit serialization_view(value_type& obj)
        requires(!is_owning)
        : base_(&obj) {}

    /// Owning constructor: takes ownership of a moved object.
    explicit serialization_view(value_type&& obj)
        requires(is_owning)
        : base_(std::move(obj)) {}

    /// Triggers deserialization if needed without returning a reference.
    /// Equivalent to `(void)**this` but expresses intent at call sites that only care
    /// about the side effect (e.g. immediately followed by `operator->`).
    void unwrap() const {
        if (needs_deserialization_)
            do_deserialize();
    }

    /// Dereference to the wrapped object, triggering deserialization if needed.
    value_type const& operator*() const {
        unwrap();
        return base_ref();
    }

    /// \overload
    value_type& operator*() {
        return const_cast<value_type&>(std::as_const(*this).operator*());
    }

    /// Arrow operator; triggers deserialization if needed.
    value_type* operator->() {
        return std::addressof(**this);
    }

    /// \overload
    value_type const* operator->() const {
        return std::addressof(**this);
    }

    // ---- Recv-side protocol -----------------------------------------------

    /// Called by infer() with the number of bytes to receive. Requires non-const T:
    /// deserialization writes into the wrapped object.
    void set_recv_count(std::ptrdiff_t n)
        requires(!std::is_const_v<value_type>)
    {
        recv_count_            = n;
        serialized_            = false;
        needs_deserialization_ = true;
        needs_resize_          = true;
    }

    // ---- MPI protocol methods --------------------------------------------

    /// Returns the number of `MPI_BYTE` elements to send or receive.
    /// On the recv side returns the count recorded by `set_recv_count()` without
    /// serializing. On the send side serializes lazily on the first call.
    std::ptrdiff_t mpi_count() const {
        if (needs_deserialization_)
            return recv_count_;
        if (!serialized_)
            do_serialize();
        return static_cast<std::ptrdiff_t>(buffer_.size());
    }

    /// Returns `MPI_BYTE` (`char`). Serialized payloads are always transported as raw bytes.
    MPI_Datatype mpi_type() const noexcept {
        return kamping::types::builtin_type<char>::data_type();
    }

    /// Send-side pointer. Serializes lazily on first call; returns `void const*` so that
    /// a const view satisfies `send_buffer` but not `recv_buffer`.
    void const* mpi_ptr() const {
        if (!needs_deserialization_ && !serialized_)
            do_serialize();
        return buffer_.data();
    }

    /// Recv-side pointer. Resizes `buffer_` to `recv_count_` bytes on the first call
    /// after `set_recv_count()`, then returns `void*` so MPI can write directly into it.
    /// A non-const view satisfies both `send_buffer` and `recv_buffer`.
    void* mpi_ptr() {
        if (needs_resize_) {
            buffer_.resize(static_cast<std::size_t>(recv_count_));
            needs_resize_ = false;
        }
        return const_cast<void*>(std::as_const(*this).mpi_ptr());
    }
};

// lvalue input (including const lvalue): non-owning.
// T deduced as U or U const → serialization_view<U&> or serialization_view<U const&>.
template <typename T>
serialization_view(T&) -> serialization_view<T&>;

// rvalue input: owning.
template <typename T>
    requires(!std::is_lvalue_reference_v<T>)
serialization_view(T&&) -> serialization_view<T>;

} // namespace kamping::v2

namespace kamping::v2::views {
/// Range adaptor that wraps any object in a `serialization_view` for MPI transport.
/// Pipe an lvalue or rvalue through it: `obj | kamping::views::serialize`.
/// Lvalues produce a non-owning view; rvalues produce an owning view.
inline constexpr struct serialize_fn : kamping::v2::adaptor_closure<serialize_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return kamping::v2::serialization_view(std::forward<R>(r));
    }
} serialize{};

/// Returns an owning serialization_view<T> with a default-constructed T.
/// Use as a recv buffer when the object does not exist yet:
///   auto view = comm.recv(kamping::v2::views::deserialize<MyType>(), 0);
///   MyType& result = *view;
template <typename T>
    requires std::default_initializable<T>
auto deserialize() {
    return kamping::v2::serialization_view<T>(T{});
}
} // namespace kamping::v2::views
