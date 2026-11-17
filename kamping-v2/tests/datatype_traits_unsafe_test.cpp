// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

// The unsafe byte-wise headers for std::pair, std::tuple and all trivially copyable types. They conflict with the
// layout-preserving headers tested in datatype_traits_test.cpp, so they get a program of their own.

#include <array>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/bcast.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/type_pool.hpp"
#include "kamping/v2/types/as_struct.hpp"
#include "kamping/v2/types/datatype_traits.hpp"
#include "kamping/v2/types/std/unsafe/trivially_copyable.hpp"
#include "kamping/v2/types/std/unsafe/tuple.hpp"
#include "kamping/v2/types/std/unsafe/utility.hpp"

namespace v2 = kamping::v2;

namespace {
struct padded {
    char   c;
    double x;
    bool   operator==(padded const&) const = default;
};

// A full specialization takes precedence over the catch-all.
struct described {
    char   c;
    double x;
};

struct not_trivially_copyable {
    std::vector<int> v;
};

int size_of(MPI_Datatype dt) {
    int size = 0;
    MPI_Type_size(dt, &size);
    return size;
}
} // namespace

template <>
struct kamping::v2::datatype_traits<described> : kamping::v2::as_struct<described> {};

static_assert(v2::has_datatype<padded>);
static_assert(v2::has_datatype<std::pair<int, double>>);
static_assert(v2::has_datatype<std::tuple<char, double>>);
static_assert(v2::has_datatype<std::array<padded, 2>>); // as_array of the bytes of padded
static_assert(!v2::has_datatype<not_trivially_copyable>);
static_assert(!v2::has_datatype<std::pair<int, not_trivially_copyable>>);

TEST(DatatypeTraitsUnsafeTest, bytes_include_the_gaps) {
    auto const padded_dt = v2::datatype_traits<padded>::datatype();
    EXPECT_EQ(size_of(padded_dt.mpi_handle()), static_cast<int>(sizeof(padded)));

    auto const pair_dt = v2::datatype_traits<std::pair<char, double>>::datatype();
    EXPECT_EQ(size_of(pair_dt.mpi_handle()), static_cast<int>(sizeof(std::pair<char, double>)));

    auto const tuple_dt = v2::datatype_traits<std::tuple<char, double>>::datatype();
    EXPECT_EQ(size_of(tuple_dt.mpi_handle()), static_cast<int>(sizeof(std::tuple<char, double>)));

    auto const described_dt = v2::datatype_traits<described>::datatype();
    EXPECT_EQ(size_of(described_dt.mpi_handle()), static_cast<int>(sizeof(char) + sizeof(double)));
}

TEST(DatatypeTraitsUnsafeTest, bcast) {
    v2::comm_view const comm = v2::comm_world();
    v2::type_pool       pool;

    std::vector<padded>                 structs(2);
    std::vector<std::pair<int, double>> pairs(2);
    if (comm.rank() == 0) {
        structs = {{'a', 0.5}, {'b', 1.5}};
        pairs   = {{1, 0.5}, {2, 1.5}};
    }
    v2::bcast(structs | v2::views::with_auto_pool(pool), MPI_COMM_WORLD);
    v2::bcast(pairs | v2::views::with_auto_pool(pool), MPI_COMM_WORLD);
    EXPECT_EQ(structs[1], (padded{'b', 1.5}));
    EXPECT_EQ(pairs[1], (std::pair<int, double>{2, 1.5}));
}
