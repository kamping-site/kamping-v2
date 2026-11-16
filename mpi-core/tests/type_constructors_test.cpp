// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <array>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/datatype.hpp"
#include "mpi/handle.hpp"
#include "mpi/type_constructors.hpp"

using namespace mpi::experimental;

namespace {
/// Sends `send` with `send_type` (count 1) to this process over MPI_COMM_SELF and receives it into a vector of
/// `recv_count` ints, i.e. returns the ints `send_type` selects from `send`, in order.
template <convertible_to_mpi_handle<MPI_Datatype> Datatype>
std::vector<int> selected_ints(int const* send, Datatype const& send_type, int recv_count) {
    std::vector<int> recv(static_cast<std::size_t>(recv_count), -1);
    int              err = MPI_Sendrecv(
        send,
        1,
        mpi::experimental::handle(send_type),
        0,
        0,
        recv.data(),
        recv_count,
        MPI_INT,
        0,
        0,
        MPI_COMM_SELF,
        MPI_STATUS_IGNORE
    );
    EXPECT_EQ(err, MPI_SUCCESS);
    return recv;
}

/// A 4 x 4 row-major matrix holding 0, 1, ..., 15.
std::array<int, 16> iota16() {
    std::array<int, 16> m{};
    std::iota(m.begin(), m.end(), 0);
    return m;
}

template <typename New>
concept contiguous_into = requires(New& n) { type_contiguous(3, MPI_INT, n); };
} // namespace

constexpr auto int_size = static_cast<MPI_Aint>(sizeof(int));

// ── Output argument ────────────────────────────────────────────────────────────────────────────────────────────────

// The new datatype is written into a raw handle or an uncommitted_datatype. A datatype is always committed, so it
// cannot receive the result of a constructor; neither can a view, which holds only a copy of the handle.
static_assert(contiguous_into<MPI_Datatype>);
static_assert(contiguous_into<uncommitted_datatype>);
static_assert(!contiguous_into<datatype>);
static_assert(!contiguous_into<datatype_view>);
static_assert(!contiguous_into<uncommitted_datatype const>);

TEST(TypeConstructorsTest, FullFormWritesRawHandle) {
    MPI_Datatype dt = MPI_DATATYPE_NULL;
    type_contiguous(3, MPI_INT, dt);
    ASSERT_NE(dt, MPI_DATATYPE_NULL);
    EXPECT_EQ(datatype_view(dt).size(), 3 * static_cast<int>(sizeof(int)));
    MPI_Type_free(&dt);
}

TEST(TypeConstructorsTest, FullFormWritesUncommittedDatatype) {
    uncommitted_datatype dt;
    type_contiguous(3, MPI_INT, dt);
    ASSERT_NE(dt.mpi_handle(), MPI_DATATYPE_NULL);
    EXPECT_EQ(dt.size(), 3 * static_cast<int>(sizeof(int)));
}

TEST(TypeConstructorsTest, ErrorThrowsAndLeavesOutputEmpty) {
    uncommitted_datatype dt;
    MPI_Comm_set_errhandler(MPI_COMM_SELF, MPI_ERRORS_RETURN);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    EXPECT_THROW(type_contiguous(-1, MPI_INT, dt), mpi_error);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    MPI_Comm_set_errhandler(MPI_COMM_SELF, MPI_ERRORS_ARE_FATAL);
    EXPECT_EQ(dt.mpi_handle(), MPI_DATATYPE_NULL);
}

// ── Base datatypes ─────────────────────────────────────────────────────────────────────────────────────────────────

TEST(TypeConstructorsTest, AcceptsEveryKindOfBaseDatatype) {
    auto const               pair       = type_contiguous(2, MPI_INT);          // uncommitted
    auto const               pair_ready = type_contiguous(2, MPI_INT).commit(); // committed
    datatype_view const      pair_view  = pair_ready;                           // view
    MPI_Datatype const       raw        = pair.mpi_handle();                    // raw
    std::array<int, 6> const send{0, 1, 2, 3, 4, 5};

    for (auto const& dt:
         {type_contiguous(3, pair).commit(),
          type_contiguous(3, pair_ready).commit(),
          type_contiguous(3, pair_view).commit(),
          type_contiguous(3, raw).commit()}) {
        EXPECT_EQ(selected_ints(send.data(), dt, 6), (std::vector<int>{0, 1, 2, 3, 4, 5}));
    }
}

TEST(TypeConstructorsTest, TemporaryBaseDatatypeIsFreedAfterTheCall) {
    // The vector is only a building block: it is freed at the end of the full expression, which MPI allows.
    auto const column = type_create_resized(type_vector(4, 1, 4, MPI_INT), 0, int_size).commit();
    EXPECT_EQ(column.extent(), int_size);

    auto const m = iota16();
    EXPECT_EQ(selected_ints(m.data(), column, 4), (std::vector<int>{0, 4, 8, 12}));
    EXPECT_EQ(selected_ints(m.data() + 1, column, 4), (std::vector<int>{1, 5, 9, 13}));
}

// ── Constructors ───────────────────────────────────────────────────────────────────────────────────────────────────

TEST(TypeConstructorsTest, Contiguous) {
    auto const dt = type_contiguous(3, MPI_INT).commit();
    EXPECT_EQ(dt.size(), 3 * static_cast<int>(sizeof(int)));
    EXPECT_EQ(dt.extent(), 3 * int_size);
    std::array<int, 3> const send{4, 5, 6};
    EXPECT_EQ(selected_ints(send.data(), dt, 3), (std::vector<int>{4, 5, 6}));
}

TEST(TypeConstructorsTest, Vector) {
    // Blocks of two ints, starting every three ints.
    auto const dt = type_vector(3, 2, 3, MPI_INT).commit();
    EXPECT_EQ(dt.size(), 6 * static_cast<int>(sizeof(int)));
    EXPECT_EQ(dt.extent(), 8 * int_size);
    auto const m = iota16();
    EXPECT_EQ(selected_ints(m.data(), dt, 6), (std::vector<int>{0, 1, 3, 4, 6, 7}));
}

TEST(TypeConstructorsTest, Hvector) {
    auto const dt = type_create_hvector(3, 2, 3 * int_size, MPI_INT).commit();
    EXPECT_EQ(dt.extent(), 8 * int_size);
    auto const m = iota16();
    EXPECT_EQ(selected_ints(m.data(), dt, 6), (std::vector<int>{0, 1, 3, 4, 6, 7}));
}

TEST(TypeConstructorsTest, Resized) {
    auto const dt = type_create_resized(MPI_INT, -int_size, 4 * int_size);
    EXPECT_EQ(dt.lower_bound(), -int_size);
    EXPECT_EQ(dt.extent(), 4 * int_size);
    EXPECT_EQ(dt.size(), static_cast<int>(sizeof(int)));
}

TEST(TypeConstructorsTest, Struct) {
    struct record {
        int    id;
        double value;
        int    tag;
    };
    std::array const              blocklengths{1, 1, 1};
    std::array<MPI_Aint, 3> const displacements{
        static_cast<MPI_Aint>(offsetof(record, id)),
        static_cast<MPI_Aint>(offsetof(record, value)),
        static_cast<MPI_Aint>(offsetof(record, tag))
    };
    std::array const types{MPI_INT, MPI_DOUBLE, MPI_INT};

    auto const record_type = type_create_resized(
                                 type_create_struct(blocklengths, displacements, types),
                                 0,
                                 static_cast<MPI_Aint>(sizeof(record))
    )
                                 .commit();
    EXPECT_EQ(record_type.size(), static_cast<int>(2 * sizeof(int) + sizeof(double)));
    EXPECT_EQ(record_type.extent(), static_cast<MPI_Aint>(sizeof(record)));

    std::array<record, 2> const send{record{1, 1.5, 10}, record{2, 2.5, 20}};
    std::array<record, 2>       recv{};
    ASSERT_EQ(
        MPI_Sendrecv(
            send.data(),
            2,
            record_type.mpi_handle(),
            0,
            0,
            recv.data(),
            2,
            record_type.mpi_handle(),
            0,
            0,
            MPI_COMM_SELF,
            MPI_STATUS_IGNORE
        ),
        MPI_SUCCESS
    );
    EXPECT_EQ(recv[1].id, 2);
    EXPECT_EQ(recv[1].value, 2.5);
    EXPECT_EQ(recv[1].tag, 20);
}

TEST(TypeConstructorsTest, Subarray) {
    // The 2 x 2 interior of a 4 x 4 matrix.
    std::array const sizes{4, 4};
    std::array const subsizes{2, 2};
    std::array const starts{1, 1};
    auto const       m = iota16();

    auto const row_major = type_create_subarray(sizes, subsizes, starts, ArrayOrder::c, MPI_INT).commit();
    EXPECT_EQ(row_major.extent(), 16 * int_size);
    EXPECT_EQ(selected_ints(m.data(), row_major, 4), (std::vector<int>{5, 6, 9, 10}));

    // Rows 0..1 and columns 1..3 in row-major order; in column-major order, sizes and starts are read the other way.
    std::array const wide{2, 3};
    std::array const offset{0, 1};
    auto const       c_order = type_create_subarray(sizes, wide, offset, ArrayOrder::c, MPI_INT).commit();
    EXPECT_EQ(selected_ints(m.data(), c_order, 6), (std::vector<int>{1, 2, 3, 5, 6, 7}));
    auto const f_order = type_create_subarray(sizes, wide, offset, ArrayOrder::fortran, MPI_INT).commit();
    EXPECT_EQ(selected_ints(m.data(), f_order, 6), (std::vector<int>{4, 5, 8, 9, 12, 13}));
}

TEST(TypeConstructorsTest, Darray) {
    // An 8-element array in blocks over the processes of MPI_COMM_WORLD; at most 8 processes.
    int size = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ASSERT_LE(size, 8);
    ASSERT_EQ(8 % size, 0);

    std::array const gsizes{8};
    std::array const distribs{MPI_DISTRIBUTE_BLOCK};
    std::array const dargs{MPI_DISTRIBUTE_DFLT_DARG};
    std::array const psizes{size};
    auto const dt = type_create_darray(size, rank, gsizes, distribs, dargs, psizes, ArrayOrder::c, MPI_INT).commit();

    int const block = 8 / size;
    EXPECT_EQ(dt.size(), block * static_cast<int>(sizeof(int)));
    EXPECT_EQ(dt.extent(), 8 * int_size);

    std::array<int, 8> global{};
    std::iota(global.begin(), global.end(), 0);
    std::vector<int> expected(static_cast<std::size_t>(block));
    std::iota(expected.begin(), expected.end(), rank * block);
    EXPECT_EQ(selected_ints(global.data(), dt, block), expected);
}

TEST(TypeConstructorsTest, Dup) {
    auto const original = type_contiguous(3, MPI_INT).commit();
    auto const copy     = type_dup(original).commit();
    EXPECT_NE(copy.mpi_handle(), original.mpi_handle());
    EXPECT_EQ(copy.size(), original.size());
    std::array<int, 3> const send{7, 8, 9};
    EXPECT_EQ(selected_ints(send.data(), copy, 3), (std::vector<int>{7, 8, 9}));

    MPI_Datatype raw = MPI_DATATYPE_NULL;
    type_dup(MPI_INT, raw);
    EXPECT_EQ(datatype_view(raw).size(), static_cast<int>(sizeof(int)));
    MPI_Type_free(&raw);
}
