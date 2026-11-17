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
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/buffer.hpp"
#include "kamping/v2/datatype.hpp"
#include "kamping/v2/handle.hpp"
#include "kamping/v2/rank.hpp"

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

// Ranges of integers, but not of booleans or characters, are accepted as displacements.
static_assert(mpi::experimental::sized_integer_range<std::vector<std::size_t>>);
static_assert(mpi::experimental::sized_integer_range<std::list<int> const&>);
static_assert(!mpi::experimental::sized_integer_range<std::vector<bool>>);
static_assert(!mpi::experimental::sized_integer_range<std::vector<char>>);
static_assert(!mpi::experimental::sized_integer_range<std::vector<double>>);

TEST(V2TypeConstructorsTest, BuiltinDatatype) {
    EXPECT_EQ(v2::builtin_datatype<double>(), MPI_DOUBLE);
    EXPECT_EQ(v2::builtin_datatype<int const&>(), MPI_INT);
    auto const dt = v2::type_vector(2, 1, 2, v2::builtin_datatype<int>()).commit();
    auto const m  = iota16();
    EXPECT_EQ(selected_ints(m.data(), dt, 2), (std::vector<int>{0, 2}));
}

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

TEST(V2TypeConstructorsTest, DarrayOfAnotherRank) {
    // Every process builds the datatypes of all blocks of an 8-element array distributed in blocks; the block of rank
    // r holds the elements r * 8 / size, ...
    int const size = world_size();
    ASSERT_EQ(8 % size, 0);
    int const local = 8 / size;

    std::array<int, 8> global{};
    std::iota(global.begin(), global.end(), 0);
    for (int r = 0; r < size; ++r) {
        auto const dt =
            v2::type_create_darray({{8, v2::distribution::block, size}}, MPI_INT, r, MPI_COMM_WORLD).commit();
        std::vector<int> expected(static_cast<std::size_t>(local));
        std::iota(expected.begin(), expected.end(), r * local);
        EXPECT_EQ(selected_ints(global.data(), dt, local), expected);
    }

    // Without a rank, the block of the calling process; with order and a rank type, the same.
    auto const own = v2::type_create_darray({{8, v2::distribution::block, size}}, MPI_INT, MPI_COMM_WORLD).commit();
    std::vector<v2::darray_dim> const dims{{.global_size = 8, .distribution = v2::distribution::block, .procs = size}};
    auto const                        typed =
        v2::type_create_darray(dims, v2::array_order::c, MPI_INT, v2::periodic_rank(MPI_COMM_WORLD), MPI_COMM_WORLD)
            .commit();
    std::vector<int> expected(static_cast<std::size_t>(local));
    std::iota(expected.begin(), expected.end(), world_rank() * local);
    EXPECT_EQ(selected_ints(global.data(), own, local), expected);
    EXPECT_EQ(selected_ints(global.data(), typed, local), expected);
}

TEST(V2TypeConstructorsTest, DarrayCyclicWithBlockSize) {
    // A 16-element array dealt out round-robin in blocks of two elements.
    int const size = world_size();
    int const rank = world_rank();
    ASSERT_EQ(16 % (2 * size), 0);

    std::vector<v2::darray_dim> const dims{
        {.global_size = 16, .distribution = v2::distribution::cyclic, .procs = size, .block_size = 2}
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

// ── type_create_struct / struct_builder ───────────────────────────────────────────────────────────────────────────

namespace {
/// Sends `n` elements of `send` to this process over MPI_COMM_SELF and receives them into `recv`, both with `dt`.
template <typename T>
void roundtrip(T const* send, T* recv, int n, MPI_Datatype dt) {
    int err = MPI_Sendrecv(send, n, dt, 0, 0, recv, n, dt, 0, 0, MPI_COMM_SELF, MPI_STATUS_IGNORE);
    EXPECT_EQ(err, MPI_SUCCESS);
}

struct vec2 {
    float x, y;
};

struct base {
    int tag = 0;
};

struct particle : base {
    char                 kind = 0;
    double               pos[3]{};
    std::array<short, 2> flags{};
    vec2                 velocity{};
    std::vector<int>     neighbors; // not sent
};

/// A class with private members and no default constructor, which describes itself.
class account {
public:
    account(int id, double balance) : _id(id), _balance(balance) {}
    int id() const {
        return _id;
    }
    double balance() const {
        return _balance;
    }
    static v2::uncommitted_datatype mpi_datatype() {
        account const sample(0, 0.0);
        return v2::struct_builder<account>(sample).field(&account::_id).field(&account::_balance).build();
    }

private:
    int    _id;
    double _balance;
};
} // namespace

// Only builtin members are deduced: other members need a datatype, arrays need array_field.
template <typename Member>
concept deduced_field = requires(v2::struct_builder<particle>& b, Member m) { b.field(m); };
template <typename Member>
concept deduced_array_field = requires(v2::struct_builder<particle>& b, Member m) { b.array_field(m); };
static_assert(deduced_field<char particle::*>);
static_assert(deduced_field<int base::*>);
static_assert(!deduced_field<vec2 particle::*>);
static_assert(!deduced_field<double (particle::*)[3]>);
static_assert(deduced_array_field<double (particle::*)[3]>);
static_assert(deduced_array_field<std::array<short, 2> particle::*>);
static_assert(!deduced_array_field<char particle::*>);
// Members of unrelated classes are rejected.
static_assert(!deduced_field<float vec2::*>);
// A builder owns datatypes: move-only.
static_assert(std::is_nothrow_move_constructible_v<v2::struct_builder<particle>>);
static_assert(!std::is_copy_constructible_v<v2::struct_builder<particle>>);

TEST(V2TypeConstructorsTest, StructFromDescriptors) {
    struct record {
        int    id;
        double value[2];
    };
    auto const dt =
        v2::type_create_resized(
            v2::type_create_struct({{offsetof(record, id), 1, MPI_INT}, {offsetof(record, value), 2, MPI_DOUBLE}}),
            0,
            sizeof(record)
        )
            .commit();
    EXPECT_EQ(dt.size(), static_cast<int>(sizeof(int) + 2 * sizeof(double)));
    std::array<record, 2> const send{{{1, {1.5, 2.5}}, {2, {3.5, 4.5}}}};
    std::array<record, 2>       recv{};
    roundtrip(send.data(), recv.data(), 2, dt.mpi_handle());
    EXPECT_EQ(recv[1].id, 2);
    EXPECT_EQ(recv[1].value[1], 4.5);
}

TEST(V2TypeConstructorsTest, StructBuilderSendsTheChosenMembers) {
    auto const vec2_type = v2::struct_builder<vec2>{}.field(&vec2::x).field(&vec2::y).build();
    auto const dt        = v2::struct_builder<particle>{}
                               .field(&particle::tag) // a member of the base class
                               .field(&particle::kind)
                               .array_field(&particle::pos)
                               .array_field(&particle::flags)
                               .field(&particle::velocity, vec2_type)
                               .build()
                               .commit();
    EXPECT_EQ(dt.extent(), static_cast<MPI_Aint>(sizeof(particle)));
    EXPECT_EQ(
        dt.size(),
        static_cast<int>(sizeof(int) + 1 + 3 * sizeof(double) + 2 * sizeof(short) + 2 * sizeof(float))
    );

    std::array<particle, 2> send;
    for (std::size_t i = 0; i < 2; ++i) {
        auto const k      = static_cast<int>(i);
        send[i].tag       = k + 1;
        send[i].kind      = static_cast<char>('a' + k);
        send[i].pos[2]    = 3.0 * k;
        send[i].flags[1]  = static_cast<short>(7 + k);
        send[i].velocity  = {static_cast<float>(k), 2.0f * static_cast<float>(k)};
        send[i].neighbors = {1, 2, 3};
    }
    std::array<particle, 2> recv;
    roundtrip(send.data(), recv.data(), 2, dt.mpi_handle());
    EXPECT_EQ(recv[1].tag, 2);
    EXPECT_EQ(recv[1].kind, 'b');
    EXPECT_EQ(recv[1].pos[2], 3.0);
    EXPECT_EQ(recv[1].flags[1], 8);
    EXPECT_EQ(recv[1].velocity.y, 2.0f);
    EXPECT_TRUE(recv[1].neighbors.empty());
}

TEST(V2TypeConstructorsTest, StructBuilderOwnsTemporaryDatatypes) {
    // The datatype of `velocity` is a temporary that the builder keeps alive until it is destroyed; the builder is
    // used in separate statements and moved.
    v2::struct_builder<particle> b;
    b.field(&particle::velocity, v2::type_contiguous(2, MPI_FLOAT));
    b.array_field(&particle::pos, v2::type_contiguous(1, MPI_DOUBLE));
    auto       moved = std::move(b);
    auto const dt    = moved.build().commit();
    EXPECT_EQ(dt.size(), static_cast<int>(2 * sizeof(float) + 3 * sizeof(double)));

    particle send;
    send.velocity = {4.0f, 5.0f};
    send.pos[1]   = 6.0;
    particle recv;
    roundtrip(&send, &recv, 1, dt.mpi_handle());
    EXPECT_EQ(recv.velocity.y, 5.0f);
    EXPECT_EQ(recv.pos[1], 6.0);
}

TEST(V2TypeConstructorsTest, StructBuilderFromSample) {
    auto const                   dt = account::mpi_datatype().commit();
    std::array<account, 2> const send{account(1, 10.5), account(2, 20.5)};
    std::array<account, 2>       recv{account(0, 0.0), account(0, 0.0)};
    roundtrip(send.data(), recv.data(), 2, dt.mpi_handle());
    EXPECT_EQ(recv[1].id(), 2);
    EXPECT_EQ(recv[1].balance(), 20.5);
}

TEST(V2TypeConstructorsTest, StructBuilderFieldOfSample) {
    // Generic code visits the members of a sample object, as as_struct does with Boost.PFR.
    vec2                     sample{};
    v2::struct_builder<vec2> b(sample);
    b.field_of_sample(sample.x, MPI_FLOAT).field_of_sample(sample.y, MPI_FLOAT);
    auto const dt = b.build().commit();

    std::array<vec2, 2> const send{{{1.0f, 2.0f}, {3.0f, 4.0f}}};
    std::array<vec2, 2>       recv{};
    roundtrip(send.data(), recv.data(), 2, dt.mpi_handle());
    EXPECT_EQ(recv[1].x, 3.0f);
    EXPECT_EQ(recv[1].y, 4.0f);
}
