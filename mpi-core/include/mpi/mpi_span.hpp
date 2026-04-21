#pragma once

#include <span>

#include <mpi.h>

#include "mpi/buffer.hpp"

/// @file
/// Concrete minimal buffer types for callers who want to talk directly to the
/// core MPI wrappers without going through the view pipeline.
///
/// Non-variadic pair (scalar count):
///   mpi_cspan  — satisfies send_buffer  (void const* ptr)
///   mpi_span   — satisfies send_buffer and recv_buffer (void* ptr); converts implicitly to mpi_cspan
///
/// Variadic pair (per-rank counts + displacements, no scalar count):
///   mpi_cspan_v — satisfies send_buffer_v
///   mpi_span_v  — satisfies send_buffer_v and recv_buffer_v; converts implicitly to mpi_cspan_v
///
/// Naming mirrors std::span / std::span<const T>: the 'c' prefix denotes a read-only view.

namespace mpi::experimental {

// ─── mpi_cspan ────────────────────────────────────────────────────────────────

/// Read-only non-owning view over a contiguous MPI buffer.
/// Satisfies send_buffer.
struct mpi_cspan {
    /// Construct from raw parts.
    constexpr mpi_cspan(void const* ptr, std::ptrdiff_t size, MPI_Datatype type) noexcept
        : _ptr(ptr), _size(size), _type(type) {}

    /// Construct from any send_buffer — deduces ptr, count, and MPI type.
    template <send_buffer Buf>
    explicit mpi_cspan(Buf const& buf) noexcept
        : _ptr(mpi::experimental::ptr(buf))
        , _size(static_cast<std::ptrdiff_t>(mpi::experimental::count(buf)))
        , _type(mpi::experimental::type(buf)) {}

    void const*    mpi_ptr()   const noexcept { return _ptr; }
    std::ptrdiff_t mpi_count() const noexcept { return _size; }
    MPI_Datatype   mpi_type()  const noexcept { return _type; }

private:
    void const*    _ptr;
    std::ptrdiff_t _size;
    MPI_Datatype   _type;
};

// ─── mpi_span ─────────────────────────────────────────────────────────────────

/// Mutable non-owning view over a contiguous MPI buffer.
/// Satisfies send_buffer and recv_buffer. Converts implicitly to mpi_cspan.
struct mpi_span {
    /// Construct from raw parts.
    constexpr mpi_span(void* ptr, std::ptrdiff_t size, MPI_Datatype type) noexcept
        : _ptr(ptr), _size(size), _type(type) {}

    /// Construct from any recv_buffer (mutable lvalue) — deduces ptr, count, and MPI type.
    template <recv_buffer Buf>
    explicit mpi_span(Buf& buf) noexcept
        : _ptr(mpi::experimental::ptr(buf))
        , _size(static_cast<std::ptrdiff_t>(mpi::experimental::count(buf)))
        , _type(mpi::experimental::type(buf)) {}

    /// Implicit conversion to mpi_cspan — analogous to T* → T const*.
    operator mpi_cspan() const noexcept { return {_ptr, _size, _type}; }

    void*          mpi_ptr()   const noexcept { return _ptr; }
    std::ptrdiff_t mpi_count() const noexcept { return _size; }
    MPI_Datatype   mpi_type()  const noexcept { return _type; }

private:
    void*          _ptr;
    std::ptrdiff_t _size;
    MPI_Datatype   _type;
};

// ─── mpi_cspan_v ──────────────────────────────────────────────────────────────

/// Read-only non-owning variadic MPI buffer view (per-rank counts + displacements).
/// Satisfies send_buffer_v. Does not carry a scalar element count.
struct mpi_cspan_v {
    /// Construct from raw parts.
    constexpr mpi_cspan_v(
        void const* ptr, MPI_Datatype type, std::span<int const> counts, std::span<int const> displs
    ) noexcept
        : _ptr(ptr), _type(type), _counts(counts), _displs(displs) {}

    /// Construct from any send_buffer plus explicit counts and displacements spans.
    /// Allows: mpi_cspan_v{data_vec, counts_vec, displs_vec}
    template <send_buffer Buf>
    mpi_cspan_v(Buf const& buf, std::span<int const> counts, std::span<int const> displs) noexcept
        : _ptr(mpi::experimental::ptr(buf))
        , _type(mpi::experimental::type(buf))
        , _counts(counts)
        , _displs(displs) {}

    /// Construct from any send_buffer_v — deduces all fields from the variadic buffer.
    template <send_buffer_v Buf>
    explicit mpi_cspan_v(Buf const& buf) noexcept
        : _ptr(mpi::experimental::ptr(buf))
        , _type(mpi::experimental::type(buf))
        , _counts(mpi::experimental::counts(buf))
        , _displs(mpi::experimental::displs(buf)) {}

    void const*          mpi_ptr()    const noexcept { return _ptr; }
    MPI_Datatype         mpi_type()   const noexcept { return _type; }
    std::span<int const> mpi_counts() const noexcept { return _counts; }
    std::span<int const> mpi_displs() const noexcept { return _displs; }

private:
    void const*          _ptr;
    MPI_Datatype         _type;
    std::span<int const> _counts;
    std::span<int const> _displs;
};

// ─── mpi_span_v ───────────────────────────────────────────────────────────────

/// Mutable non-owning variadic MPI buffer view (per-rank counts + displacements).
/// Satisfies send_buffer_v and recv_buffer_v. Converts implicitly to mpi_cspan_v.
struct mpi_span_v {
    /// Construct from raw parts.
    constexpr mpi_span_v(
        void* ptr, MPI_Datatype type, std::span<int const> counts, std::span<int const> displs
    ) noexcept
        : _ptr(ptr), _type(type), _counts(counts), _displs(displs) {}

    /// Construct from any recv_buffer (mutable lvalue) plus explicit counts and displacements spans.
    /// Allows: mpi_span_v{data_vec, counts_vec, displs_vec}
    template <recv_buffer Buf>
    mpi_span_v(Buf& buf, std::span<int const> counts, std::span<int const> displs) noexcept
        : _ptr(mpi::experimental::ptr(buf))
        , _type(mpi::experimental::type(buf))
        , _counts(counts)
        , _displs(displs) {}

    /// Construct from any recv_buffer_v (mutable lvalue) — deduces all fields.
    template <recv_buffer_v Buf>
    explicit mpi_span_v(Buf& buf) noexcept
        : _ptr(mpi::experimental::ptr(buf))
        , _type(mpi::experimental::type(buf))
        , _counts(mpi::experimental::counts(buf))
        , _displs(mpi::experimental::displs(buf)) {}

    /// Implicit conversion to mpi_cspan_v — analogous to T* → T const*.
    operator mpi_cspan_v() const noexcept { return {_ptr, _type, _counts, _displs}; }

    void*                mpi_ptr()    const noexcept { return _ptr; }
    MPI_Datatype         mpi_type()   const noexcept { return _type; }
    std::span<int const> mpi_counts() const noexcept { return _counts; }
    std::span<int const> mpi_displs() const noexcept { return _displs; }

private:
    void*                _ptr;
    MPI_Datatype         _type;
    std::span<int const> _counts;
    std::span<int const> _displs;
};

} // namespace mpi::experimental
