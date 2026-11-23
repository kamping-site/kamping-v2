// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <chrono>
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

// A strong typedef sent as its representation.
struct meters {
    double value;
};

// Declared to be sent as a double, but too large: not a builtin type.
struct padded_meters {
    double value;
    int    tag;
};

// Declared to be sent as a double, but not trivially copyable: not a builtin type.
struct tracked_meters {
    double value;
    tracked_meters(tracked_meters const& other) : value(other.value) {}
};

// Declared to be sent as an enumeration, which is not an arithmetic type: not a builtin type.
struct colored {
    color value;
};
} // namespace

template <>
struct mpi::experimental::representation_traits<meters> {
    using type = double;
};

template <>
struct mpi::experimental::representation_traits<padded_meters> {
    using type = double;
};

template <>
struct mpi::experimental::representation_traits<tracked_meters> {
    using type = double;
};

template <>
struct mpi::experimental::representation_traits<colored> {
    using type = color;
};

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
static_assert(builtin_mpi_type<char8_t>);
static_assert(builtin_mpi_type<char16_t>);
static_assert(builtin_mpi_type<char32_t>);

// ── representation_traits ──────────────────────────────────────────────────────────────────────────────────────────

static_assert(builtin_mpi_type<std::chrono::nanoseconds>);
static_assert(builtin_mpi_type<std::chrono::duration<double> const&>);
static_assert(builtin_mpi_type<std::chrono::duration<float, std::milli>>);
static_assert(builtin_mpi_type<std::chrono::duration<unsigned, std::micro>>);
static_assert(builtin_mpi_type<meters>);
static_assert(!builtin_mpi_type<padded_meters>);
static_assert(!builtin_mpi_type<tracked_meters>);
static_assert(!builtin_mpi_type<colored>);
// The period is not part of the representation; time points are not sent.
static_assert(!builtin_mpi_type<std::chrono::sys_seconds>);

static_assert(send_buffer<std::vector<std::chrono::milliseconds>>);
static_assert(recv_buffer<std::vector<meters>>);

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

TEST(BuiltinTypesTest, unicode_characters_use_unsigned_integers_of_their_size) {
    EXPECT_EQ(builtin_datatype<char8_t>(), MPI_UNSIGNED_CHAR);
    EXPECT_EQ(builtin_datatype<char16_t>(), builtin_datatype<std::uint_least16_t>());
    EXPECT_EQ(builtin_datatype<char32_t>(), builtin_datatype<std::uint_least32_t>());
}

TEST(BuiltinTypesTest, represented_classes_use_the_datatype_of_their_representation) {
    EXPECT_EQ(builtin_datatype<std::chrono::duration<double>>(), MPI_DOUBLE);
    EXPECT_EQ(builtin_datatype<std::chrono::nanoseconds>(), builtin_datatype<std::chrono::nanoseconds::rep>());
    EXPECT_EQ(builtin_datatype<meters>(), MPI_DOUBLE);
    std::vector<std::chrono::milliseconds> times(3);
    EXPECT_EQ(mpi::experimental::type(times), builtin_datatype<std::chrono::milliseconds::rep>());
    EXPECT_EQ(mpi::experimental::count(times), 3);
}

TEST(BuiltinTypesTest, durations_round_trip_through_mpi) {
    using namespace std::chrono_literals;
    std::vector<std::chrono::nanoseconds> const sent{1ns, -2s, std::chrono::nanoseconds::max()};
    std::vector<std::chrono::nanoseconds>       received(3);
    int                                         err = MPI_Sendrecv(
        sent.data(),
        3,
        builtin_datatype<std::chrono::nanoseconds>(),
        0,
        0,
        received.data(),
        3,
        builtin_datatype<std::chrono::nanoseconds>(),
        0,
        0,
        MPI_COMM_SELF,
        MPI_STATUS_IGNORE
    );
    ASSERT_EQ(err, MPI_SUCCESS);
    EXPECT_EQ(received, sent);
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
