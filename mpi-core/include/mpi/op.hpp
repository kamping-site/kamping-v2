// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/buffer.hpp"
#include "mpi/builtin_types.hpp"
#include "mpi/datatype.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"
#include "mpi/mpi_span.hpp"
#include "mpi/ops.hpp"

/// @file
/// Reduction operation objects: the owning handle `op`, created by `make_op` from an MPI callback, the non-owning
/// `op_view`, and `functor_op`, created by `make_op` from a C++ functor; `reduce_local` (`MPI_Reduce_local`), which
/// applies an operation locally. `op` and `op_view` are callable on single values through `reduce_local`.

namespace mpi::experimental {

/// @brief The type of `commutative`, which marks a reduction operation as commutative.
struct commutative_tag {};
/// @brief The type of `non_commutative`, which marks a reduction operation as non-commutative.
struct non_commutative_tag {};

/// @brief Marks a reduction operation as commutative, which allows MPI to combine the values in any order.
inline constexpr commutative_tag commutative{};
/// @brief Marks a reduction operation as non-commutative: MPI combines the values in rank order.
inline constexpr non_commutative_tag non_commutative{};

/// @brief One of the commutativity tags, `commutative_tag` or `non_commutative_tag`.
template <typename C>
concept commutativity_tag = std::same_as<C, commutative_tag> || std::same_as<C, non_commutative_tag>;

/// @brief The signature of the callback of a user-defined reduction operation, as taken by `MPI_Op_create`: it
/// combines `*len` elements of datatype `*datatype` from `invec` into `inoutvec`.
using user_function = void(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype);

/// @brief A functor that can combine elements of type `T` as MPI reduction operation.
///
/// `F` is default-constructible, i.e. stateless, and callable as `T(T const&, T const&)`; `T` is destructible and
/// move-constructible. MPI calls the operation through a plain function, which uses a default-constructed `F`.
template <typename F, typename T>
concept reduction_functor = std::default_initializable<F> && std::is_invocable_r_v<T, F, T const&, T const&>
                            && std::destructible<T> && std::move_constructible<T>;

namespace detail {
// The MPI callback of make_op: combines invec[i] and inoutvec[i] with a default-constructed F and stores the result in
// inoutvec[i]. The result is constructed in place after destroying the old element, so T need not be assignable, e.g.
// std::pair<K const, V>. An exception cannot propagate through MPI, so it terminates the program.
template <typename T, typename F>
void apply_elementwise(void* invec, void* inoutvec, int* len, MPI_Datatype* /* datatype */) noexcept {
    T* in    = static_cast<T*>(invec);
    T* inout = static_cast<T*>(inoutvec);
    for (int i = 0; i < *len; ++i) {
        T combined = F{}(in[i], inout[i]);
        std::destroy_at(inout + i);
        std::construct_at(inout + i, std::move(combined));
    }
}
} // namespace detail

/// @brief Combines the elements of `inbuf` into `inoutbuf` with `op`: `inoutbuf[i] = inbuf[i] op inoutbuf[i]`
/// (`MPI_Reduce_local`).
///
/// No communication is involved; the operation is applied by the calling process.
///
/// @param inbuf The first operands.
/// @param inoutbuf The second operands, overwritten with the results.
/// @param op The operation: an `MPI_Op`, an operation object such as `op` or `functor_op`, or a functor of a
/// predefined operation such as `std::plus<>`.
/// @pre `count(inbuf) == count(inoutbuf)` and `type(inbuf) == type(inoutbuf)`.
/// @throws mpi_error if `MPI_Reduce_local` fails.
template <send_buffer SBuf, recv_buffer RBuf, valid_op<SBuf, RBuf> Op>
void reduce_local(SBuf&& inbuf, RBuf&& inoutbuf, Op const& op) {
    using count_t = decltype(mpi::experimental::count(inbuf));
    KAMPING_V2_ASSERT(
        mpi::experimental::count(inbuf) == static_cast<count_t>(mpi::experimental::count(inoutbuf)),
        "reduce_local: both buffers must have the same count"
    );
    KAMPING_V2_ASSERT(
        mpi::experimental::type(inbuf) == mpi::experimental::type(inoutbuf),
        "reduce_local: both buffers must have the same type"
    );
    int err = MPI_Reduce_local(
        mpi::experimental::ptr(inbuf),
        mpi::experimental::ptr(inoutbuf),
        static_cast<int>(mpi::experimental::count(inoutbuf)),
        mpi::experimental::type(inoutbuf),
        mpi::experimental::as_mpi_op(op, inbuf, inoutbuf)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <typename T, typename F>
    requires reduction_functor<F, T>
class functor_op;

/// CRTP mixin that makes an operation handle callable on single values: `o(a, b)` returns `a op b`, computed with
/// `reduce_local`. `Derived` must implement `MPI_Op mpi_handle() const`.
///
/// Each call is an MPI call, so an operation created from a C++ functor is faster to apply with the functor itself, as
/// `functor_op` does.
template <typename Derived>
class op_accessors {
public:
    /// @return `lhs op rhs` for values of a builtin type, e.g. `op_view{MPI_SUM}(1, 2) == 3`.
    /// @throws mpi_error if `MPI_Reduce_local` fails.
    template <builtin_mpi_type T>
    [[nodiscard]] T operator()(T const& lhs, T const& rhs) const {
        return (*this)(lhs, rhs, mpi::experimental::builtin_datatype<T>());
    }

    /// @return `lhs op rhs`, with `lhs` and `rhs` described by `dt`.
    /// @pre `dt` describes one `T` and is committed.
    /// @throws mpi_error if `MPI_Reduce_local` fails.
    template <std::copy_constructible T>
    [[nodiscard]] T operator()(T const& lhs, T const& rhs, datatype_view dt) const {
        T result = rhs;
        mpi::experimental::reduce_local(
            mpi_cspan(std::addressof(lhs), 1, dt.mpi_handle()),
            mpi_span(std::addressof(result), 1, dt.mpi_handle()),
            static_cast<Derived const&>(*this)
        );
        return result;
    }
};

/// @brief Non-owning wrapper around an `MPI_Op`.
///
/// Implicitly constructible from a raw `MPI_Op`, including the predefined ones such as `MPI_SUM`, and from an `op` or
/// `functor_op`, so non-template code can take an `op_view` to accept any operation. It never frees the operation.
/// Like `op`, it is callable on single values.
class op_view : public op_accessors<op_view> {
public:
    /// @brief Views `o`, which must outlive the view.
    op_view(MPI_Op o) noexcept : _op(o) {}

    /// @return The underlying `MPI_Op`.
    [[nodiscard]] MPI_Op mpi_handle() const noexcept {
        return _op;
    }

private:
    MPI_Op _op;
};

/// @brief Owning RAII wrapper for a user-defined `MPI_Op`.
///
/// Created by `make_op` from an MPI callback, or moved from a `functor_op`, and frees the operation on
/// destruction. It can be passed as the operation of the reduce-like collectives, and is callable on single values (see
/// `op_accessors`). Predefined operations such as `MPI_SUM` are used as raw `MPI_Op`, `op_view` or functors such as
/// `std::plus<>`, never owned by an `op`. Move-only.
///
/// @code
/// auto const max_abs = make_op<int>([](int a, int b) { return std::max(std::abs(a), std::abs(b)); }, commutative);
/// allreduce(local, global, max_abs, comm);
/// @endcode
class op : public op_accessors<op> {
public:
    /// @brief Holds no operation (`MPI_OP_NULL`).
    op() noexcept = default;

    op(op const&)            = delete;
    op& operator=(op const&) = delete;

    /// @brief Transfers ownership; `o` holds `MPI_OP_NULL` afterwards.
    op(op&& o) noexcept : _op(std::exchange(o._op, MPI_OP_NULL)) {}

    /// @brief Takes over the operation of `f`, which holds `MPI_OP_NULL` afterwards; the functor is dropped.
    template <typename T, typename F>
    op(functor_op<T, F>&& f) noexcept : _op(f.release()) {}

    /// @brief Frees the current operation, then transfers ownership; `o` holds `MPI_OP_NULL` afterwards.
    op& operator=(op&& o) noexcept {
        if (this != &o) {
            free_if_valid();
            _op = std::exchange(o._op, MPI_OP_NULL);
        }
        return *this;
    }

    /// @brief Frees the operation, unless it is `MPI_OP_NULL`.
    ~op() noexcept {
        free_if_valid();
    }

    /// @brief Takes ownership of the user-defined operation `o`.
    /// @pre `o` is `MPI_OP_NULL` or was created with `MPI_Op_create`; a predefined operation such as `MPI_SUM` must not
    /// be freed.
    [[nodiscard]] static op from_native(MPI_Op o) noexcept {
        return op(o);
    }

    /// @brief Frees the operation now; afterwards `*this` holds `MPI_OP_NULL`.
    ///
    /// Use this instead of the destructor to handle errors of `MPI_Op_free`. The handle is reset before the MPI call,
    /// so the destructor never frees it a second time.
    /// @throws mpi_error if `MPI_Op_free` fails.
    void free() {
        if (_op != MPI_OP_NULL) {
            MPI_Op tmp = std::exchange(_op, MPI_OP_NULL);
            int    err = MPI_Op_free(&tmp);
            if (err != MPI_SUCCESS) {
                throw mpi_error(err);
            }
        }
    }

    /// @brief Gives up ownership; `*this` holds `MPI_OP_NULL` afterwards.
    /// @return The operation, which the caller has to free with `MPI_Op_free`.
    [[nodiscard]] MPI_Op release() noexcept {
        return std::exchange(_op, MPI_OP_NULL);
    }

    /// @return The underlying `MPI_Op`.
    [[nodiscard]] MPI_Op mpi_handle() const noexcept {
        return _op;
    }

    /// @brief Views the operation, which must outlive the view.
    operator op_view() const& noexcept {
        return op_view{_op};
    }
    /// A view of a temporary would dangle as soon as the temporary frees its operation.
    operator op_view() const&& = delete;

    /// @return A pointer to the underlying `MPI_Op`, so that `MPI_Op_create` can write the new operation into
    /// `*this`.
    /// @pre `*this` holds `MPI_OP_NULL`. Writing into it otherwise would leak the operation it holds.
    [[nodiscard]] MPI_Op* mpi_handle_ptr() noexcept {
        KAMPING_V2_ASSERT(
            _op == MPI_OP_NULL,
            "op::mpi_handle_ptr: the operation must be empty, or the operation it holds is leaked"
        );
        return &_op;
    }

private:
    explicit op(MPI_Op o) noexcept : _op(o) {}

    void free_if_valid() noexcept {
        if (_op != MPI_OP_NULL) {
            MPI_Op_free(&_op);
            _op = MPI_OP_NULL;
        }
    }

    MPI_Op _op = MPI_OP_NULL;
};

/// @brief Creates a user-defined reduction operation from an MPI callback (`MPI_Op_create`).
///
/// The callback is already type-erased, so no element type is given. To create an operation from a C++ functor, use
/// `make_op<T>(f)`.
///
/// @code
/// void sum_ints(void* in, void* inout, int* len, MPI_Datatype*) { ... }
/// auto const sum = make_op(&sum_ints, commutative);
/// @endcode
///
/// @param function The callback, which combines the elements of its first argument into its second.
/// @param commutativity `commutative` or `non_commutative` (the default).
/// @return The new operation, owned.
/// @throws mpi_error if `MPI_Op_create` fails.
template <commutativity_tag Commutativity = non_commutative_tag>
[[nodiscard]] op make_op(user_function* function, Commutativity /* commutativity */ = {}) {
    KAMPING_V2_ASSERT(function != nullptr, "make_op: the callback must not be null");
    op  result;
    int err = MPI_Op_create(function, std::same_as<Commutativity, commutative_tag> ? 1 : 0, result.mpi_handle_ptr());
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
    return result;
}

/// @brief A user-defined reduction operation on elements of type `T`, created from the functor `F`, as returned by
/// `make_op`.
///
/// It owns its `MPI_Op` like an `op` and can be passed as the operation of the reduce-like collectives. It is also
/// callable as `T(T const&, T const&)`, which calls the functor directly, so the same object combines values locally,
/// e.g. in `std::reduce`, without an MPI call. Moving it into an `op` keeps the operation and drops the functor.
/// Move-only.
template <typename T, typename F>
    requires reduction_functor<F, T>
class functor_op {
public:
    /// @brief Creates the operation for `f` (`MPI_Op_create`).
    ///
    /// MPI combines elements with a default-constructed `F`; `f` is kept for `operator()`. `T` need not be assignable:
    /// MPI's callback replaces an element by destroying and constructing it in place. MPI passes the elements as raw
    /// memory, so `T` must be sent with a datatype that transfers its whole value representation.
    /// @param f The functor.
    /// @param commutativity `commutative` or `non_commutative`.
    /// @throws mpi_error if `MPI_Op_create` fails.
    template <commutativity_tag Commutativity>
    functor_op(F f, Commutativity commutativity)
        : _f(std::move(f)),
          _op(mpi::experimental::make_op(&detail::apply_elementwise<T, F>, commutativity)) {}

    /// @return `f(lhs, rhs)` with the functor.
    T operator()(T const& lhs, T const& rhs) const {
        return _f(lhs, rhs);
    }

    /// @brief Frees the operation now; afterwards it holds `MPI_OP_NULL`, but can still be called.
    /// @throws mpi_error if `MPI_Op_free` fails.
    void free() {
        _op.free();
    }

    /// @brief Gives up ownership of the operation; afterwards it holds `MPI_OP_NULL`, but can still be called.
    /// @return The operation, which the caller has to free with `MPI_Op_free`.
    [[nodiscard]] MPI_Op release() noexcept {
        return _op.release();
    }

    /// @return The underlying `MPI_Op`.
    [[nodiscard]] MPI_Op mpi_handle() const noexcept {
        return _op.mpi_handle();
    }

    /// @brief Views the operation, which must outlive the view.
    operator op_view() const& noexcept {
        return op_view{_op.mpi_handle()};
    }
    /// A view of a temporary would dangle as soon as the temporary frees its operation.
    operator op_view() const&& = delete;

private:
    F  _f;
    op _op;
};

/// @brief Creates a user-defined reduction operation on elements of type `T` from a binary functor.
///
/// The functor must be default-constructible, i.e. stateless, since MPI calls the operation through a plain function:
/// a struct with `operator()` or a lambda without captures. `T` need not be assignable: the result replaces an element
/// by destroying and constructing it in place. MPI passes the elements as raw memory, so `T` must be sent with a
/// datatype that transfers its whole value representation. To create an operation from an MPI callback, use
/// `make_op(callback)`.
///
/// The result owns the operation and is also callable with the functor, so it can combine values locally as well:
///
/// @code
/// auto const sum = make_op<int>([](int a, int b) { return a + b; });   // non-commutative
/// int const local = std::reduce(v.begin(), v.end(), 0, std::cref(sum));
/// allreduce(views::ref_single(local), global, sum, comm);
/// auto const max_op = make_op<int>(max_functor{}, commutative);
/// @endcode
///
/// @tparam T The element type, given explicitly: `make_op<int>(...)`.
/// @param f The functor, callable as `T(T const&, T const&)`.
/// @param commutativity `commutative` or `non_commutative` (the default).
/// @return The new operation, owned.
/// @throws mpi_error if `MPI_Op_create` fails.
template <typename T, typename F, commutativity_tag Commutativity = non_commutative_tag>
    requires reduction_functor<std::remove_cvref_t<F>, T>
[[nodiscard]] functor_op<T, std::remove_cvref_t<F>> make_op(F&& f, Commutativity commutativity = {}) {
    return functor_op<T, std::remove_cvref_t<F>>(std::forward<F>(f), commutativity);
}

} // namespace mpi::experimental
