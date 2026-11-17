// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <complex>
#include <cstddef>
#include <functional>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/datatype.hpp"
#include "mpi/handle.hpp"
#include "mpi/mpi_span.hpp"
#include "mpi/op.hpp"
#include "mpi/ops.hpp"

using namespace mpi::experimental;

namespace {
enum class color { red, green };

// Sums ints element-wise.
void int_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* /* datatype */) {
    auto const* in    = static_cast<int const*>(invec);
    auto*       inout = static_cast<int*>(inoutvec);
    for (int i = 0; i < *len; ++i) {
        inout[i] += in[i];
    }
}

// An element type that can be copied but not assigned, like the value type of a std::map.
struct pinned {
    int const value;
};

struct pinned_max {
    pinned operator()(pinned const& a, pinned const& b) const {
        return a.value >= b.value ? a : b;
    }
};

int comm_size() {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

int comm_rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}
} // namespace

// ── op handle ──────────────────────────────────────────────────────────────────────────────────────────────────────

static_assert(convertible_to_mpi_handle<op, MPI_Op>);
static_assert(convertible_to_mpi_handle_ptr<op, MPI_Op>);
static_assert(!std::is_copy_constructible_v<op>);
static_assert(std::is_nothrow_move_constructible_v<op>);
static_assert(std::is_nothrow_move_assignable_v<op>);

// op_view accepts raw handles and lvalues of the owning types, but no temporaries, which would dangle.
static_assert(convertible_to_mpi_handle<op_view, MPI_Op>);
static_assert(std::is_convertible_v<MPI_Op, op_view>);
static_assert(std::is_convertible_v<op const&, op_view>);
static_assert(!std::is_convertible_v<op&&, op_view>);
using int_sum_op = functor_op<int, std::plus<>>;
static_assert(std::is_convertible_v<int_sum_op const&, op_view>);
static_assert(!std::is_convertible_v<int_sum_op&&, op_view>);

// A functor_op moves into an op, dropping its functor; an lvalue does not convert, so ownership never moves silently.
static_assert(convertible_to_mpi_handle<int_sum_op, MPI_Op>);
static_assert(!std::is_copy_constructible_v<int_sum_op>);
static_assert(std::is_nothrow_move_constructible_v<int_sum_op>);
static_assert(std::is_convertible_v<int_sum_op&&, op>);
static_assert(!std::is_convertible_v<int_sum_op&, op>);
static_assert(std::same_as<decltype(make_op<int>(std::plus<>{})), int_sum_op>);
static_assert(std::same_as<decltype(make_op(&int_sum)), op>);

// The operation objects are callable on single values.
static_assert(std::is_invocable_r_v<int, op const&, int, int>);
static_assert(std::is_invocable_r_v<double, op_view, double, double>);
static_assert(!std::is_invocable_v<op_view, int*, int*>);
static_assert(std::is_invocable_r_v<int, int_sum_op const&, int, int>);

// ── builtin_op ─────────────────────────────────────────────────────────────────────────────────────────────────────

static_assert(builtin_op<std::plus<>, int>);
static_assert(builtin_op<std::plus<int>, int const&>);
static_assert(!builtin_op<std::plus<long>, int>);
static_assert(builtin_op<std::plus<>, std::complex<double>>);
static_assert(builtin_op<ops::max<>, double>);
static_assert(!builtin_op<ops::max<>, std::complex<double>>);
static_assert(!builtin_op<std::minus<>, int>);
static_assert(builtin_op<std::bit_and<>, std::byte>);
static_assert(!builtin_op<std::bit_and<>, double>);
static_assert(builtin_op<ops::logical_xor<>, bool>);
static_assert(builtin_op<std::logical_and<>, unpacked_bool>);
static_assert(!builtin_op<std::plus<>, bool>);
static_assert(!builtin_op<std::plus<>, char>); // MPI_CHAR is not in the integer group
// Enumerations are sent as their underlying type, but have no predefined operations.
static_assert(builtin_mpi_type<color>);
static_assert(!builtin_op<std::plus<>, color>);
static_assert(!builtin_op<ops::max<>, color>);
// ops:: names the standard functors as well.
static_assert(std::is_same_v<ops::plus<>, std::plus<>>);
static_assert(valid_op<std::plus<>, std::vector<int>, std::vector<int>>);
static_assert(!valid_op<std::plus<>, std::vector<color>, std::vector<color>>);
static_assert(valid_op<op, std::vector<color>, std::vector<color>>);

// reduction_op: callable locally and an MPI operation on the element type.
static_assert(reduction_op<std::plus<>, int>);
static_assert(reduction_op<ops::max<>, double const&>);
static_assert(reduction_op<functor_op<int, std::plus<>>, int>);
static_assert(reduction_op<op, int>);
static_assert(reduction_op<op_view, double>);
static_assert(reduction_op<op, pinned>); // applied locally through reduce_local, given the datatype
static_assert(!reduction_op<std::minus<>, int>);
static_assert(!reduction_op<std::plus<>, color>);
static_assert(reduction_op<MPI_Op, int>); // likewise, with the builtin datatype
static_assert(!reduction_op<decltype([](int a, int b) { return a + b; }), int>);

TEST(OpTest, builtin_op_handle) {
    EXPECT_EQ((builtin_op_handle<std::plus<>, int>()), MPI_SUM);
    EXPECT_EQ((builtin_op_handle<std::multiplies<>, float>()), MPI_PROD);
    EXPECT_EQ((builtin_op_handle<ops::max<>, double>()), MPI_MAX);
    EXPECT_EQ((builtin_op_handle<ops::min<int>, int>()), MPI_MIN);
    EXPECT_EQ((builtin_op_handle<std::logical_and<>, bool>()), MPI_LAND);
    EXPECT_EQ((builtin_op_handle<std::logical_or<>, bool>()), MPI_LOR);
    EXPECT_EQ((builtin_op_handle<ops::logical_xor<>, bool>()), MPI_LXOR);
    EXPECT_EQ((builtin_op_handle<std::bit_and<>, unsigned>()), MPI_BAND);
    EXPECT_EQ((builtin_op_handle<std::bit_or<>, std::byte>()), MPI_BOR);
    EXPECT_EQ((builtin_op_handle<std::bit_xor<>, long>()), MPI_BXOR);
}

TEST(OpTest, functors) {
    EXPECT_EQ(ops::max<>{}(1, 3), 3);
    EXPECT_EQ(ops::min<int>{}(1, 3), 1);
    EXPECT_TRUE(ops::logical_xor<>{}(true, 0));
    EXPECT_FALSE(ops::logical_xor<bool>{}(true, true));
}

TEST(OpTest, make_op_from_callback) {
    auto const sum    = make_op(&int_sum, commutative);
    int const  local  = comm_rank();
    int        global = 0;
    int        err    = MPI_Allreduce(&local, &global, 1, MPI_INT, sum.mpi_handle(), MPI_COMM_WORLD);
    ASSERT_EQ(err, MPI_SUCCESS);
    EXPECT_EQ(global, comm_size() * (comm_size() - 1) / 2);
}

TEST(OpTest, make_op_from_lambda) {
    auto const sum    = make_op<int>([](int a, int b) { return a + b; });
    int const  local  = comm_rank() + 1;
    int        global = 0;
    int        err    = MPI_Allreduce(&local, &global, 1, MPI_INT, sum.mpi_handle(), MPI_COMM_WORLD);
    ASSERT_EQ(err, MPI_SUCCESS);
    EXPECT_EQ(global, comm_size() * (comm_size() + 1) / 2);
}

TEST(OpTest, make_op_does_not_assign_elements) {
    // One pinned is one int; a derived datatype makes MPI call the operation once per pinned.
    MPI_Datatype pinned_type = MPI_DATATYPE_NULL;
    int          err         = MPI_Type_contiguous(1, MPI_INT, &pinned_type);
    ASSERT_EQ(err, MPI_SUCCESS);
    err = MPI_Type_commit(&pinned_type);
    ASSERT_EQ(err, MPI_SUCCESS);

    auto const                max_op = make_op<pinned>(pinned_max{}, commutative);
    std::vector<pinned> const local{{comm_rank()}, {-comm_rank()}};
    std::vector<int>          global(2, -1);
    err = MPI_Allreduce(local.data(), global.data(), 2, pinned_type, max_op.mpi_handle(), MPI_COMM_WORLD);
    ASSERT_EQ(err, MPI_SUCCESS);
    EXPECT_EQ(global[0], comm_size() - 1);
    EXPECT_EQ(global[1], 0);

    MPI_Type_free(&pinned_type);
}

TEST(OpTest, ownership) {
    op empty;
    EXPECT_EQ(empty.mpi_handle(), MPI_OP_NULL);

    op sum = make_op(&int_sum); // non-commutative by default
    ASSERT_NE(sum.mpi_handle(), MPI_OP_NULL);
    int commute = 1;
    int err     = MPI_Op_commutative(sum.mpi_handle(), &commute);
    ASSERT_EQ(err, MPI_SUCCESS);
    EXPECT_EQ(commute, 0);

    op moved = std::move(sum);
    EXPECT_EQ(sum.mpi_handle(), MPI_OP_NULL); // NOLINT(bugprone-use-after-move)
    EXPECT_NE(moved.mpi_handle(), MPI_OP_NULL);

    MPI_Op raw = moved.release();
    EXPECT_EQ(moved.mpi_handle(), MPI_OP_NULL);
    op adopted = op::from_native(raw);
    EXPECT_EQ(adopted.mpi_handle(), raw);

    adopted.free();
    EXPECT_EQ(adopted.mpi_handle(), MPI_OP_NULL);
    adopted.free(); // no-op on MPI_OP_NULL
}

TEST(OpTest, functor_op_moves_into_op) {
    op o = make_op<int>(std::plus<>{}, commutative);
    ASSERT_NE(o.mpi_handle(), MPI_OP_NULL);
    int const local  = 1;
    int       global = 0;
    int       err    = MPI_Allreduce(&local, &global, 1, MPI_INT, o.mpi_handle(), MPI_COMM_WORLD);
    ASSERT_EQ(err, MPI_SUCCESS);
    EXPECT_EQ(global, comm_size());

    std::vector<op> ops;
    ops.push_back(make_op<int>(std::multiplies<>{}));
    auto sum = make_op<int>(std::plus<>{});
    ops.emplace_back(std::move(sum));
    EXPECT_EQ(sum.mpi_handle(), MPI_OP_NULL); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(sum(2, 3), 5);                  // the functor is still there
    EXPECT_EQ(ops[0](3, 4), 12);              // through MPI_Reduce_local
    EXPECT_EQ(ops[1](3, 4), 7);
}

TEST(OpTest, call_operator_uses_reduce_local) {
    op_view const sum = MPI_SUM;
    EXPECT_EQ(sum(1, 2), 3);
    EXPECT_DOUBLE_EQ(op_view{MPI_MAX}(1.5, -2.0), 1.5);

    auto const custom = make_op(&int_sum, commutative);
    EXPECT_EQ(custom(20, 22), 42);
    op_view const view = custom;
    EXPECT_EQ(view(1, 1), 2);

    std::vector<int> const values{1, 2, 3, 4};
    EXPECT_EQ(std::reduce(values.begin(), values.end(), 0, std::cref(custom)), 10);
}

TEST(OpTest, call_operator_with_explicit_datatype) {
    // One pinned is one int.
    MPI_Datatype pinned_type = MPI_DATATYPE_NULL;
    int          err         = MPI_Type_contiguous(1, MPI_INT, &pinned_type);
    ASSERT_EQ(err, MPI_SUCCESS);
    err = MPI_Type_commit(&pinned_type);
    ASSERT_EQ(err, MPI_SUCCESS);

    op const     max_op = make_op<pinned>(pinned_max{}, commutative);
    pinned const result = max_op(pinned{3}, pinned{7}, pinned_type);
    EXPECT_EQ(result.value, 7);

    MPI_Type_free(&pinned_type);
}

TEST(OpTest, reduce_local) {
    std::vector<int> const in{1, 2, 3};
    std::vector<int>       inout{10, 20, 30};
    mpi::experimental::reduce_local(in, inout, std::plus<>{});
    EXPECT_EQ(inout, (std::vector<int>{11, 22, 33}));

    mpi::experimental::reduce_local(in, inout, MPI_MAX);
    EXPECT_EQ(inout, (std::vector<int>{11, 22, 33}));

    // Non-commutative: inout[i] = in[i] op inout[i].
    auto const keep_first = make_op<int>([](int a, int /* b */) { return a; });
    mpi::experimental::reduce_local(in, inout, keep_first);
    EXPECT_EQ(inout, in);
}

TEST(OpTest, as_local_op) {
    // A callable operation is applied directly.
    std::plus<> const plus;
    auto const        local_plus = as_local_op<int>(plus);
    static_assert(std::same_as<std::remove_const_t<decltype(local_plus)>, std::reference_wrapper<std::plus<> const>>);
    EXPECT_EQ(local_plus(1, 2), 3);

    // An MPI_Op is applied through MPI_Reduce_local.
    EXPECT_EQ(as_local_op<int>(MPI_PROD)(3, 4), 12);
    std::vector<int> const values{1, 2, 3, 4};
    EXPECT_EQ(std::reduce(values.begin(), values.end(), 0, as_local_op<int>(MPI_SUM)), 10);

    // A derived type, with its datatype.
    MPI_Datatype pinned_type = MPI_DATATYPE_NULL;
    int          err         = MPI_Type_contiguous(1, MPI_INT, &pinned_type);
    ASSERT_EQ(err, MPI_SUCCESS);
    err = MPI_Type_commit(&pinned_type);
    ASSERT_EQ(err, MPI_SUCCESS);
    op const     max_op = make_op<pinned>(pinned_max{}, commutative);
    pinned const result = as_local_op<pinned>(max_op, pinned_type)(pinned{3}, pinned{7});
    EXPECT_EQ(result.value, 7);
    MPI_Type_free(&pinned_type);
}
