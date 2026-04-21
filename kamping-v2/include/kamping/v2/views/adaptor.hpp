#pragma once

/// @file
/// Range adaptor machinery for kamping views.
///
/// This file provides the building blocks for pipe-able view factories (e.g. `with_type`,
/// `with_counts`). The design mirrors the C++23 range adaptor pattern but extends it to
/// non-range types (MPI buffers, scalars, etc.).
///
/// ## Types and their roles
///
///   adaptor_closure_base        Tag base class; any type derived from it is pipe-able.
///
///   adaptor_closure<Derived>    CRTP base that injects `operator|` overloads via ADL:
///                                 val | closure        →  closure(val)
///                                 closure1 | closure2  →  composed_closure{c1, c2}
///
///   composed_closure<F, S>      Result of composing two closures. Itself a closure, so
///                               arbitrary chains work: (c1 | c2) | c3.
///
///   partial_adaptor<Fn, Args>   A closure holding a function and pre-bound arguments.
///                               Created by partial application of an `adaptor`.
///                               Invoking it (directly or via |) prepends the incoming
///                               value and calls the stored function.
///
///   adaptor<N, Fn>              Factory with arity-based disambiguation:
///                                 N args       →  partial application (returns partial_adaptor)
///                                 N+1 args     →  full call (first arg is the value)
///
/// ## End-to-end example
///
///   // Declaration (in with_type_view.hpp):
///   inline constexpr adaptor<1, decltype([](auto&& r, MPI_Datatype t) {
///       return with_type_view(all(std::forward<decltype(r)>(r)), t);
///   })> with_type{};
///
///   // Usage:
///   vec | with_type(MPI_INT)      // partial → partial_adaptor holding {fn, MPI_INT}
///                                 //         → operator| calls fn(vec, MPI_INT)
///                                 //         → returns with_type_view{ref_view{vec}, MPI_INT}
///
///   with_type(vec, MPI_INT)       // full call → fn(vec, MPI_INT) directly
///
///   auto pipe = with_type(MPI_INT) | with_counts(c);   // composed_closure{partial1, partial2}
///   vec | pipe                                          // applies left-to-right

#include <cstddef>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

#include "kamping/v2/views/all.hpp"

namespace kamping::v2 {

// ──────────────────────────────────────────────────────────────────────────────
// Closure base and concept
// ──────────────────────────────────────────────────────────────────────────────

/// Tag base class — any type inheriting (transitively) from this is recognized as a pipe-able closure.
struct adaptor_closure_base {};

/// Checks whether T is a pipe-able closure (i.e., something produced by partial application of an adaptor).
template <typename T>
concept is_adaptor_closure = std::derived_from<std::remove_cvref_t<T>, adaptor_closure_base>;

/// Detects external range adaptor closures (e.g. std::views::take(2)) by duck typing:
/// they are not ranges themselves but are callable with a range argument and produce a range.
/// Used to route std_closure | kamping_closure to composition rather than applying the
/// kamping closure to the std closure as a value.
template <typename T>
concept is_external_closure = !std::ranges::range<std::remove_cvref_t<T>> && !is_adaptor_closure<T>
                              && requires(std::remove_cvref_t<T> const& t, std::ranges::empty_view<int> r) {
                                     { t(r) } -> std::ranges::range;
                                 };

// ──────────────────────────────────────────────────────────────────────────────
// adaptor_closure — CRTP base providing operator|
// ──────────────────────────────────────────────────────────────────────────────

template <typename First, typename Second>
struct composed_closure;

/// CRTP base that makes Derived pipe-able. Provides two operator| overloads found via ADL:
///   - val | closure        → closure(val)            when val is NOT itself a closure
///   - closure1 | closure2  → composed_closure{...}   when both sides are closures
/// The !is_adaptor_closure constraint on the left side is intentionally the *only* constraint,
/// so any value type (range, MPI buffer, scalar, ...) can appear on the left side of |.
template <typename Derived>
struct adaptor_closure : adaptor_closure_base {
    // val | closure  →  closure(val)
    template <typename T>
        requires(!is_adaptor_closure<T>) && (!is_external_closure<T>)
    friend constexpr auto operator|(T&& val, Derived const& self) {
        return self(std::forward<T>(val));
    }

    // val | rvalue_closure  →  std::move(closure)(val)
    // Moves stored arguments out of temporary closures so that views take ownership
    // rather than holding a dangling ref_view into the closure's storage.
    template <typename T>
        requires(!is_adaptor_closure<T>) && (!is_external_closure<T>)
    friend constexpr auto operator|(T&& val, Derived&& self) {
        return std::move(self)(std::forward<T>(val));
    }

    // closure | closure  →  composed_closure that applies them left-to-right
    template <typename Other>
        requires is_adaptor_closure<Other> || is_external_closure<Other>
    friend constexpr auto operator|(Other other, Derived self) {
        return composed_closure<Other, Derived>(std::move(other), std::move(self));
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// composed_closure — result of closure1 | closure2
// ──────────────────────────────────────────────────────────────────────────────

/// The result of closure1 | closure2. Itself a closure, so chains of arbitrary length work:
///   (c1 | c2) | c3  →  composed_closure<composed_closure<C1,C2>, C3>
template <typename First, typename Second>
struct composed_closure : adaptor_closure<composed_closure<First, Second>> {
    [[no_unique_address]] First  first_;
    [[no_unique_address]] Second second_;

    constexpr composed_closure(First first, Second second) : first_(std::move(first)), second_(std::move(second)) {}

    template <typename T>
    constexpr auto operator()(T&& val) const& {
        return second_(first_(std::forward<T>(val)));
    }

    // Rvalue overload: moves stored closures so that bound arguments are transferred
    // into the resulting view rather than copied from the about-to-be-destroyed composition.
    template <typename T>
    constexpr auto operator()(T&& val) && {
        return std::move(second_)(std::move(first_)(std::forward<T>(val)));
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// partial_adaptor — closure holding a function and pre-bound arguments
// ──────────────────────────────────────────────────────────────────────────────

namespace detail {
/// Wraps a single argument for storage in a partial_adaptor:
///   - range arguments are wrapped via all() (lvalue → ref_view, rvalue → owning_view)
///   - non-range, non-copyable lvalue references are stored as std::reference_wrapper
///     (handles move-only types such as type_pool that must not be copied)
///   - all other non-range arguments (copyable types like MPI_Datatype, int, …) are stored by value (std::decay_t)
template <typename Arg>
constexpr decltype(auto) store_arg(Arg&& arg) {
    if constexpr (std::ranges::range<std::remove_cvref_t<Arg>>)
        return kamping::v2::all(std::forward<Arg>(arg));
    else if constexpr (std::is_lvalue_reference_v<Arg>
                       && !std::is_copy_constructible_v<std::remove_reference_t<Arg>>)
        return std::ref(arg);
    else
        return std::decay_t<Arg>(std::forward<Arg>(arg));
}
} // namespace detail

/// A closure with pre-bound arguments. Created by calling an adaptor with only the extra arguments
/// (partial application). When invoked (directly or via |), prepends the incoming value and calls fn_.
///
/// Example: with_type(MPI_INT) returns a partial_adaptor holding fn_ and MPI_INT in bound_.
///          vec | that_closure  →  fn_(vec, MPI_INT)  →  with_type_view(vec, MPI_INT)
template <typename Fn, typename... BoundArgs>
struct partial_adaptor : adaptor_closure<partial_adaptor<Fn, BoundArgs...>> {
    [[no_unique_address]] Fn fn_;
    std::tuple<BoundArgs...> bound_;

    constexpr partial_adaptor(Fn fn, std::tuple<BoundArgs...> bound) : fn_(std::move(fn)), bound_(std::move(bound)) {}

    template <typename T>
    constexpr auto operator()(T&& val) const& {
        return std::apply([&](auto const&... args) { return fn_(std::forward<T>(val), args...); }, bound_);
    }

    // Rvalue overload: moves stored arguments out so the resulting view takes ownership
    // rather than holding a ref_view into this closure's (about-to-be-destroyed) storage.
    template <typename T>
    constexpr auto operator()(T&& val) && {
        return std::apply([&](auto&&... args) { return fn_(std::forward<T>(val), std::move(args)...); }, bound_);
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// adaptor — factory with arity-based partial/full call disambiguation
// ──────────────────────────────────────────────────────────────────────────────

/// Generic range adaptor factory, parameterized by the number of extra arguments (beyond the value).
/// Arity-based disambiguation (like libstdc++ internally):
///   - ExtraArgs arguments       → partial application, returns a pipeable partial_adaptor
///   - ExtraArgs + 1 arguments   → full call, first argument is the value
///
/// Usage:  inline constexpr adaptor<1, decltype([](auto&& r, MPI_Datatype t) { ... })> with_type{};
///         vec | with_type(MPI_INT)       // partial → partial_adaptor → pipe applies it
///         with_type(vec, MPI_INT)        // full call
template <std::size_t ExtraArgs, typename Fn>
struct adaptor {
    [[no_unique_address]] Fn fn_;

    /// Partial application: bind ExtraArgs arguments, return a pipeable closure.
    /// Range arguments are stored via all_t — lvalue ranges become ref_view (borrow the original),
    /// rvalue ranges become owning_view. Non-range arguments are stored by value (std::decay_t).
    template <typename... Args>
        requires(sizeof...(Args) == ExtraArgs)
    constexpr auto operator()(Args&&... args) const {
        return partial_adaptor<Fn, decltype(detail::store_arg(std::forward<Args>(args)))...>(
            fn_,
            std::tuple{detail::store_arg(std::forward<Args>(args))...}
        );
    }

    /// Full call: first argument is the value, remaining ExtraArgs are forwarded to fn_.
    template <typename T, typename... Args>
        requires(sizeof...(Args) == ExtraArgs)
    constexpr auto operator()(T&& val, Args&&... args) const {
        return fn_(std::forward<T>(val), std::forward<Args>(args)...);
    }
};

} // namespace kamping::v2
