// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <numeric>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/datatype.hpp"
#include "kamping/v2/handle.hpp"

namespace v2 = kamping::v2;

namespace {
/// Sends `send` with `send_type` (count 1) to this process over MPI_COMM_SELF and receives it into a vector of
/// `recv_count` ints, i.e. returns the ints `send_type` selects from `send`, in order.
template <v2::convertible_to_mpi_handle<MPI_Datatype> Datatype>
std::vector<int> selected_ints(int const* send, Datatype const& send_type, int recv_count) {
    std::vector<int> recv(static_cast<std::size_t>(recv_count), -1);
    int              err = MPI_Sendrecv(
        send,
        1,
        v2::handle(send_type),
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

constexpr auto int_size = static_cast<MPI_Aint>(sizeof(int));

int world_size() {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

int world_rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}
} // namespace

// The descriptors are aggregates, so designated initializers work.
static_assert(std::is_aggregate_v<v2::block>);
static_assert(std::is_aggregate_v<v2::hblock>);
static_assert(std::is_aggregate_v<v2::subarray_dim>);
static_assert(std::is_aggregate_v<v2::darray_dim>);

// ── type_indexed / type_create_hindexed ────────────────────────────────────────────────────────────────────────────

TEST(V2TypeConstructorsTest, IndexedFromBracedBlocks) {
    // The upper triangle of the 4 x 4 matrix: row i holds 4 - i elements, starting at column i.
    auto const dt = v2::type_indexed({{0, 4}, {5, 3}, {10, 2}, {15, 1}}, MPI_INT).commit();
    EXPECT_EQ(dt.size(), 10 * static_cast<int>(sizeof(int)));
    auto const m = iota16();
    EXPECT_EQ(selected_ints(m.data(), dt, 10), (std::vector<int>{0, 1, 2, 3, 5, 6, 7, 10, 11, 15}));
}

TEST(V2TypeConstructorsTest, IndexedFromSpanOfBlocks) {
    std::vector<v2::block> const blocks{{.displacement = 3, .length = 2}, {.displacement = 9, .length = 1}};
    auto const                   dt = v2::type_indexed(blocks, MPI_INT).commit();
    auto const                   m  = iota16();
    EXPECT_EQ(selected_ints(m.data(), dt, 3), (std::vector<int>{3, 4, 9}));
}

TEST(V2TypeConstructorsTest, Hindexed) {
    auto const dt = v2::type_create_hindexed({{3 * int_size, 2}, {9 * int_size, 1}}, MPI_INT).commit();
    auto const m  = iota16();
    EXPECT_EQ(selected_ints(m.data(), dt, 3), (std::vector<int>{3, 4, 9}));

    std::vector<v2::hblock> const blocks{{12 * int_size, 1}};
    auto const                    single = v2::type_create_hindexed(blocks, MPI_INT).commit();
    EXPECT_EQ(selected_ints(m.data(), single, 1), (std::vector<int>{12}));
}

TEST(V2TypeConstructorsTest, IndexedEmpty) {
    EXPECT_EQ(v2::type_indexed({}, MPI_INT).size(), 0);
    EXPECT_EQ(v2::type_create_hindexed(std::span<v2::hblock const>{}, MPI_INT).size(), 0);
}

TEST(V2TypeConstructorsTest, CoreFormsStayAvailable) {
    std::array const blocklengths{2, 1};
    std::array const displacements{3, 9};
    auto const       dt = v2::type_indexed(blocklengths, displacements, MPI_INT).commit();
    auto const       m  = iota16();
    EXPECT_EQ(selected_ints(m.data(), dt, 3), (std::vector<int>{3, 4, 9}));
}

// ── type_create_indexed_block / type_create_hindexed_block ─────────────────────────────────────────────────────────

TEST(V2TypeConstructorsTest, IndexedBlockFromAnyIntegerRange) {
    auto const m = iota16();

    std::vector<std::size_t> const sizes{0, 5, 7};
    auto const                     from_size_t = v2::type_create_indexed_block(2, sizes, MPI_INT).commit();
    EXPECT_EQ(selected_ints(m.data(), from_size_t, 6), (std::vector<int>{0, 1, 5, 6, 7, 8}));

    std::list<std::int64_t> const non_contiguous{0, 5, 7};
    auto const                    from_list = v2::type_create_indexed_block(2, non_contiguous, MPI_INT).commit();
    EXPECT_EQ(selected_ints(m.data(), from_list, 6), (std::vector<int>{0, 1, 5, 6, 7, 8}));

    std::vector<int> const ints{0, 5, 7};
    auto const             from_ints = v2::type_create_indexed_block(2, ints, MPI_INT).commit();
    EXPECT_EQ(selected_ints(m.data(), from_ints, 6), (std::vector<int>{0, 1, 5, 6, 7, 8}));

    auto const from_view = v2::type_create_indexed_block(1, std::views::iota(0, 3), MPI_INT).commit();
    EXPECT_EQ(selected_ints(m.data(), from_view, 3), (std::vector<int>{0, 1, 2}));
}

TEST(V2TypeConstructorsTest, IndexedBlockSelectsElementsByIndex) {
    // The elements are sent in the order of the indices, not in memory order.
    std::vector<std::size_t> const indices{12, 2, 0, 7};
    auto const                     dt = v2::type_create_indexed_block(indices, MPI_INT).commit();
    EXPECT_EQ(dt.size(), 4 * static_cast<int>(sizeof(int)));
    auto const m = iota16();
    EXPECT_EQ(selected_ints(m.data(), dt, 4), (std::vector<int>{12, 2, 0, 7}));
}

TEST(V2TypeConstructorsTest, HindexedBlock) {
    auto const m = iota16();

    std::vector<std::size_t> const offsets{12 * sizeof(int), 2 * sizeof(int)};
    auto const                     pairs = v2::type_create_hindexed_block(2, offsets, MPI_INT).commit();
    EXPECT_EQ(selected_ints(m.data(), pairs, 4), (std::vector<int>{12, 13, 2, 3}));

    auto const single = v2::type_create_hindexed_block(offsets, MPI_INT).commit();
    EXPECT_EQ(selected_ints(m.data(), single, 2), (std::vector<int>{12, 2}));
}

// ── type_create_subarray ───────────────────────────────────────────────────────────────────────────────────────────

TEST(V2TypeConstructorsTest, Subarray) {
    auto const m = iota16();

    // The 2 x 2 interior of the 4 x 4 matrix, row-major if no order is given.
    auto const interior = v2::type_create_subarray({{4, 1, 2}, {4, 1, 2}}, MPI_INT).commit();
    EXPECT_EQ(interior.extent(), 16 * int_size);
    EXPECT_EQ(selected_ints(m.data(), interior, 4), (std::vector<int>{5, 6, 9, 10}));

    // Rows 0..1 and columns 1..3 in row-major order; in column-major order, the dimensions are read the other way.
    std::array<v2::subarray_dim, 2> const dims{
        {{.size = 4, .start = 0, .subsize = 2}, {.size = 4, .start = 1, .subsize = 3}}
    };
    auto const c_order = v2::type_create_subarray(dims, v2::array_order::c, MPI_INT).commit();
    EXPECT_EQ(selected_ints(m.data(), c_order, 6), (std::vector<int>{1, 2, 3, 5, 6, 7}));
    auto const f_order = v2::type_create_subarray(dims, v2::array_order::fortran, MPI_INT).commit();
    EXPECT_EQ(selected_ints(m.data(), f_order, 6), (std::vector<int>{4, 5, 8, 9, 12, 13}));
    auto const f_braced = v2::type_create_subarray({{4, 0, 2}, {4, 1, 3}}, v2::array_order::fortran, MPI_INT).commit();
    EXPECT_EQ(selected_ints(m.data(), f_braced, 6), (std::vector<int>{4, 5, 8, 9, 12, 13}));
}

// ── type_create_darray ─────────────────────────────────────────────────────────────────────────────────────────────

TEST(V2TypeConstructorsTest, DarrayBlock) {
    // An 8-element array in blocks over the processes of MPI_COMM_WORLD; at most 8 processes.
    int const size = world_size();
    int const rank = world_rank();
    ASSERT_EQ(8 % size, 0);

    auto const dt    = v2::type_create_darray({{8, v2::distribution::block, size}}, MPI_INT, MPI_COMM_WORLD).commit();
    int const  local = 8 / size;
    EXPECT_EQ(dt.size(), local * static_cast<int>(sizeof(int)));
    EXPECT_EQ(dt.extent(), 8 * int_size);

    std::array<int, 8> global{};
    std::iota(global.begin(), global.end(), 0);
    std::vector<int> expected(static_cast<std::size_t>(local));
    std::iota(expected.begin(), expected.end(), rank * local);
    EXPECT_EQ(selected_ints(global.data(), dt, local), expected);
}

TEST(V2TypeConstructorsTest, DarrayCyclicWithBlockSize) {
    // A 16-element array dealt out round-robin in blocks of two elements.
    int const size = world_size();
    int const rank = world_rank();
    ASSERT_EQ(16 % (2 * size), 0);

    std::vector<v2::darray_dim> const dims{
        {.global_size = 16, .distrib = v2::distribution::cyclic, .procs = size, .block_size = 2}
    };
    auto const dt    = v2::type_create_darray(dims, v2::array_order::c, MPI_INT, MPI_COMM_WORLD).commit();
    int const  local = 16 / size;
    EXPECT_EQ(dt.size(), local * static_cast<int>(sizeof(int)));

    auto const       m = iota16();
    std::vector<int> expected;
    for (int start = 2 * rank; start < 16; start += 2 * size) {
        expected.push_back(start);
        expected.push_back(start + 1);
    }
    EXPECT_EQ(selected_ints(m.data(), dt, local), expected);
}

TEST(V2TypeConstructorsTest, DarrayWithUndistributedDimension) {
    // The rows of a size x 4 matrix, one per process; the columns are not distributed.
    int const size = world_size();
    int const rank = world_rank();

    auto const dt =
        v2::type_create_darray({{size, v2::distribution::block, size}, {4}}, MPI_INT, MPI_COMM_WORLD).commit();
    EXPECT_EQ(dt.size(), 4 * static_cast<int>(sizeof(int)));

    std::vector<int> global(static_cast<std::size_t>(4 * size));
    std::iota(global.begin(), global.end(), 0);
    std::vector<int> expected(4);
    std::iota(expected.begin(), expected.end(), 4 * rank);
    EXPECT_EQ(selected_ints(global.data(), dt, 4), expected);
}
