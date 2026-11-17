// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/builtin_types.hpp"

using namespace mpi::experimental;

namespace {
enum class color : std::uint8_t { red, green };
enum plain_enum { first, second };
struct aggregate {
    int    a;
    double b;
};
} // namespace

// ── builtin_mpi_type ───────────────────────────────────────────────────────────────────────────────────────────────

static_assert(builtin_mpi_type<int>);
static_assert(builtin_mpi_type<int const&>);
static_assert(builtin_mpi_type<double volatile>);
static_assert(builtin_mpi_type<std::byte>);
static_assert(builtin_mpi_type<bool>);
static_assert(builtin_mpi_type<unpacked_bool>);
static_assert(builtin_mpi_type<std::complex<double>>);
static_assert(builtin_mpi_type<color>);
static_assert(builtin_mpi_type<plain_enum>);
static_assert(!builtin_mpi_type<aggregate>);
static_assert(!builtin_mpi_type<int*>);
static_assert(!builtin_mpi_type<int[2]>);
static_assert(!builtin_mpi_type<std::complex<int>>);

// Contiguous ranges of builtin types are buffers; std::vector<bool> is not contiguous.
static_assert(send_buffer<std::vector<color>>);
static_assert(recv_buffer<std::vector<unpacked_bool>>);
static_assert(!data_buffer<std::vector<bool>>);

// ── unpacked_bool ──────────────────────────────────────────────────────────────────────────────────────────────────

static_assert(sizeof(unpacked_bool) == sizeof(bool));
static_assert(!unpacked_bool{});
static_assert(unpacked_bool{true});
static_assert(std::is_trivially_copyable_v<unpacked_bool>);

TEST(BuiltinTypesTest, builtin_datatype) {
    EXPECT_EQ(builtin_datatype<int>(), MPI_INT);
    EXPECT_EQ(builtin_datatype<int const&>(), MPI_INT);
    EXPECT_EQ(builtin_datatype<unsigned long long>(), MPI_UNSIGNED_LONG_LONG);
    EXPECT_EQ(builtin_datatype<char>(), MPI_CHAR);
    EXPECT_EQ(builtin_datatype<signed char>(), MPI_SIGNED_CHAR);
    EXPECT_EQ(builtin_datatype<double>(), MPI_DOUBLE);
    EXPECT_EQ(builtin_datatype<std::byte>(), MPI_BYTE);
    EXPECT_EQ(builtin_datatype<bool>(), MPI_CXX_BOOL);
    EXPECT_EQ(builtin_datatype<unpacked_bool>(), MPI_CXX_BOOL);
    EXPECT_EQ(builtin_datatype<std::complex<float>>(), MPI_CXX_FLOAT_COMPLEX);
}

TEST(BuiltinTypesTest, enums_use_the_datatype_of_their_underlying_type) {
    EXPECT_EQ(builtin_datatype<color>(), builtin_datatype<std::uint8_t>());
    EXPECT_EQ(builtin_datatype<plain_enum>(), builtin_datatype<std::underlying_type_t<plain_enum>>());
}

TEST(BuiltinTypesTest, range_buffer_type) {
    std::vector<color> colors{color::red, color::green};
    EXPECT_EQ(mpi::experimental::type(colors), builtin_datatype<std::uint8_t>());
    std::vector<unpacked_bool> flags{true, false, true};
    EXPECT_EQ(mpi::experimental::type(flags), MPI_CXX_BOOL);
    EXPECT_EQ(mpi::experimental::count(flags), 3);
}

TEST(BuiltinTypesTest, unpacked_bool_round_trip_through_mpi) {
    std::vector<unpacked_bool> const sent{true, false, true};
    std::vector<unpacked_bool>       received(3);
    int                              err = MPI_Sendrecv(
        sent.data(),
        3,
        builtin_datatype<unpacked_bool>(),
        0,
        0,
        received.data(),
        3,
        builtin_datatype<unpacked_bool>(),
        0,
        0,
        MPI_COMM_SELF,
        MPI_STATUS_IGNORE
    );
    ASSERT_EQ(err, MPI_SUCCESS);
    EXPECT_TRUE(received[0]);
    EXPECT_FALSE(received[1]);
    EXPECT_TRUE(received[2]);
}
