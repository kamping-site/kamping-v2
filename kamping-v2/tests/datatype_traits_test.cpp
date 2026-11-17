// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

// The datatypes of types described by datatype_traits, with the layout-preserving headers for std::pair and
// std::tuple. The unsafe byte-wise variants are tested in datatype_traits_unsafe_test.cpp: a program must include only
// one of the two.

#include <array>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/buffer.hpp"
#include "kamping/v2/collectives/bcast.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/type_constructors.hpp"
#include "kamping/v2/type_pool.hpp"
#include "kamping/v2/types/as_struct.hpp"
#include "kamping/v2/types/datatype_traits.hpp"
#include "kamping/v2/types/std/tuple.hpp"
#include "kamping/v2/types/std/utility.hpp"

namespace v2 = kamping::v2;

namespace {
enum class color : std::uint8_t { red, green, blue };

// Gaps after `c` and after `tag`.
struct point {
    char   c;
    double x;
    int    tag;
    bool   operator==(point const&) const = default;
};

struct vec3 {
    float x, y, z;
    bool  operator==(vec3 const&) const = default;
};

// Nested: members with derived datatypes.
struct particle {
    point                  where;
    std::array<double, 2>  velocity;
    color                  kind;
    std::pair<int, double> extra;
    bool                   operator==(particle const&) const = default;
};

// Not reflectable, and only some members are sent.
class cell {
public:
    cell() = default;
    cell(int id, double weight) : _id(id), _weight(weight) {}
    int id() const {
        return _id;
    }
    double weight() const {
        return _weight;
    }
    int cache() const {
        return _cache;
    }

    friend struct kamping::v2::datatype_traits<cell>;

private:
    int    _id     = 0;
    double _weight = 0.0;
    int    _cache  = 0; // not sent
};

struct no_datatype {
    int* ptr;
};

struct aggregate_without_trait {
    int    a;
    double b;
};

int extent_of(MPI_Datatype dt) {
    MPI_Aint lb     = 0;
    MPI_Aint extent = 0;
    MPI_Type_get_extent(dt, &lb, &extent);
    return static_cast<int>(extent);
}

int size_of(MPI_Datatype dt) {
    int size = 0;
    MPI_Type_size(dt, &size);
    return size;
}
} // namespace

template <>
struct kamping::v2::datatype_traits<point> : kamping::v2::as_struct<point> {};
template <>
struct kamping::v2::datatype_traits<vec3> : kamping::v2::as_array<float, 3> {};
template <>
struct kamping::v2::datatype_traits<particle> : kamping::v2::as_struct<particle> {};
template <>
struct kamping::v2::datatype_traits<cell> {
    static mpi::experimental::uncommitted_datatype datatype() {
        return kamping::v2::struct_builder<cell>{}.field(&cell::_id).field(&cell::_weight).build();
    }
};

// ── has_datatype ───────────────────────────────────────────────────────────────────────────────────────────────────

static_assert(v2::has_datatype<int>);
static_assert(v2::has_datatype<double const&>);
static_assert(v2::has_datatype<color>);
static_assert(v2::has_datatype<int[4]>);
static_assert(v2::has_datatype<std::array<color, 3>>);
static_assert(v2::has_datatype<std::array<point, 3>>);
static_assert(v2::has_datatype<point>);
static_assert(v2::has_datatype<cell>);
static_assert(v2::has_datatype<std::pair<int, point>>);
static_assert(v2::has_datatype<std::tuple<int, double, color>>);
static_assert(!v2::has_datatype<std::tuple<>>);
static_assert(!v2::has_datatype<std::pair<int, no_datatype>>);
static_assert(!v2::has_datatype<std::array<int, 0>>);
static_assert(!v2::has_datatype<aggregate_without_trait>); // reflection is opt-in per type
static_assert(!v2::has_datatype<int*>);

// ── the datatypes ──────────────────────────────────────────────────────────────────────────────────────────────────

TEST(DatatypeTraitsTest, as_struct_skips_the_gaps) {
    auto const dt = v2::datatype_traits<point>::datatype();
    EXPECT_EQ(extent_of(dt.mpi_handle()), static_cast<int>(sizeof(point)));
    EXPECT_EQ(size_of(dt.mpi_handle()), static_cast<int>(sizeof(char) + sizeof(double) + sizeof(int)));
}

TEST(DatatypeTraitsTest, as_array) {
    auto const dt = v2::datatype_traits<vec3>::datatype();
    EXPECT_EQ(extent_of(dt.mpi_handle()), static_cast<int>(sizeof(vec3)));
    EXPECT_EQ(size_of(dt.mpi_handle()), static_cast<int>(3 * sizeof(float)));

    auto const arr = v2::datatype_traits<std::array<point, 2>>::datatype();
    EXPECT_EQ(extent_of(arr.mpi_handle()), static_cast<int>(sizeof(std::array<point, 2>)));
}

TEST(DatatypeTraitsTest, as_bytes) {
    auto const dt = v2::as_bytes<point>::datatype();
    EXPECT_EQ(extent_of(dt.mpi_handle()), static_cast<int>(sizeof(point)));
    EXPECT_EQ(size_of(dt.mpi_handle()), static_cast<int>(sizeof(point)));
}

TEST(DatatypeTraitsTest, struct_builder_sends_only_some_members) {
    auto const dt = v2::datatype_traits<cell>::datatype();
    EXPECT_EQ(extent_of(dt.mpi_handle()), static_cast<int>(sizeof(cell)));
    EXPECT_EQ(size_of(dt.mpi_handle()), static_cast<int>(sizeof(int) + sizeof(double)));
}

TEST(DatatypeTraitsTest, pair_and_tuple_skip_the_gaps) {
    auto const pair_dt = v2::datatype_traits<std::pair<char, double>>::datatype();
    EXPECT_EQ(extent_of(pair_dt.mpi_handle()), static_cast<int>(sizeof(std::pair<char, double>)));
    EXPECT_EQ(size_of(pair_dt.mpi_handle()), static_cast<int>(sizeof(char) + sizeof(double)));

    auto const tuple_dt = v2::datatype_traits<std::tuple<int, char, double>>::datatype();
    EXPECT_EQ(extent_of(tuple_dt.mpi_handle()), static_cast<int>(sizeof(std::tuple<int, char, double>)));
    EXPECT_EQ(size_of(tuple_dt.mpi_handle()), static_cast<int>(sizeof(int) + sizeof(char) + sizeof(double)));
}

// ── sending through a pool ─────────────────────────────────────────────────────────────────────────────────────────

TEST(DatatypeTraitsTest, bcast_nested_struct) {
    v2::comm_view const comm = v2::comm_world();
    v2::type_pool       pool;

    particle const        expected{{'p', 1.5, 7}, {2.0, -3.0}, color::blue, {11, 0.25}};
    std::vector<particle> v(3);
    if (comm.rank() == 0) {
        v.assign(3, expected);
    }
    v2::bcast(v | v2::views::with_auto_pool(pool), MPI_COMM_WORLD);
    for (auto const& p: v) {
        EXPECT_EQ(p, expected);
    }
}

TEST(DatatypeTraitsTest, bcast_tuples_and_hand_written_traits) {
    v2::comm_view const comm = v2::comm_world();
    v2::type_pool       pool;

    std::vector<std::tuple<int, char, double>> tuples(2);
    std::vector<cell>                          cells(2);
    if (comm.rank() == 0) {
        tuples = {{1, 'a', 0.5}, {2, 'b', 1.5}};
        cells  = {cell{1, 0.5}, cell{2, 1.5}};
    }
    v2::bcast(tuples | v2::views::with_auto_pool(pool), MPI_COMM_WORLD);
    v2::bcast(cells | v2::views::with_auto_pool(pool), MPI_COMM_WORLD);
    EXPECT_EQ(tuples[1], (std::tuple<int, char, double>{2, 'b', 1.5}));
    EXPECT_EQ(cells[1].id(), 2);
    EXPECT_DOUBLE_EQ(cells[1].weight(), 1.5);
}

TEST(DatatypeTraitsTest, enums_are_builtin) {
    v2::comm_view const comm = v2::comm_world();
    std::vector<color>  colors(2);
    if (comm.rank() == 0) {
        colors = {color::green, color::blue};
    }
    v2::bcast(colors, MPI_COMM_WORLD); // no pool needed
    EXPECT_EQ(colors, (std::vector<color>{color::green, color::blue}));

    v2::type_pool pool;
    EXPECT_EQ(pool.register_type<color>(), v2::builtin_datatype<std::uint8_t>());
}
