#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/types/contiguous_type.hpp"
#include "kamping/types/mpi_type_traits.hpp"
#include "kamping/v2/collectives/bcast.hpp"
#include "kamping/v2/type_pool.hpp"
#include "mpi/buffer.hpp"
#include "mpi/comm.hpp"

struct MyStruct {
    int    x;
    double y;
};

template <>
struct kamping::types::mpi_type_traits<MyStruct> : public kamping::types::byte_serialized<MyStruct> {};

// Builtin type: find() always returns the predefined MPI type, even without commit().
TEST(TypePoolTest, BuiltinTypeAlwaysFound) {
    kamping::v2::type_pool pool;
    auto                   dt = pool.find<int>();
    ASSERT_TRUE(dt.has_value());
    EXPECT_EQ(*dt, MPI_INT);
}

// register_type() for a builtin is a no-op: returns the builtin type directly.
TEST(TypePoolTest, RegisterBuiltinReturnsPredefinedType) {
    kamping::v2::type_pool pool;
    EXPECT_EQ(pool.register_type<int>(), MPI_INT);
}

// Custom type: find() returns nullopt before register_type().
TEST(TypePoolTest, CustomTypeNotFoundBeforeRegister) {
    kamping::v2::type_pool pool;
    EXPECT_FALSE(pool.find<MyStruct>().has_value());
}

// register_type() registers the type; find() then returns the same handle.
TEST(TypePoolTest, RegisterThenFind) {
    kamping::v2::type_pool pool;
    MPI_Datatype           registered = pool.register_type<MyStruct>();
    ASSERT_NE(registered, MPI_DATATYPE_NULL);
    auto found = pool.find<MyStruct>();
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, registered);
}

// register_type() is idempotent: calling it twice returns the same handle.
TEST(TypePoolTest, RegisterIdempotent) {
    kamping::v2::type_pool pool;
    MPI_Datatype           first  = pool.register_type<MyStruct>();
    MPI_Datatype           second = pool.register_type<MyStruct>();
    EXPECT_EQ(first, second);
}

// with_pool adaptor attaches the registered type to the view.
TEST(TypePoolTest, WithPoolSetsCorrectType) {
    kamping::v2::type_pool pool;
    MPI_Datatype           registered = pool.register_type<MyStruct>();
    std::vector<MyStruct>  v{{1, 2.0}};
    auto                   view = v | kamping::v2::views::with_pool(pool);
    EXPECT_EQ(mpi::experimental::type(view), registered);
}

// with_auto_pool registers the type on first use; find() returns it afterwards.
TEST(TypePoolTest, WithAutoPoolRegistersType) {
    kamping::v2::type_pool pool;
    ASSERT_FALSE(pool.find<MyStruct>().has_value());
    std::vector<MyStruct> v{{1, 2.0}};
    auto                  view = v | kamping::v2::views::with_auto_pool(pool);
    ASSERT_TRUE(pool.find<MyStruct>().has_value());
    EXPECT_EQ(mpi::experimental::type(view), *pool.find<MyStruct>());
}

// Integration: bcast a vector of custom structs using with_pool.
TEST(TypePoolTest, BcastWithPool) {
    mpi::experimental::comm_view comm{MPI_COMM_WORLD};
    kamping::v2::type_pool pool;
    pool.register_type<MyStruct>();
    std::vector<MyStruct> v;
    if (comm.rank() == 0) {
        v.emplace_back(MyStruct{42, 3.14});
    } else {
        v.emplace_back(MyStruct{0, 0.0});
    }
    kamping::v2::bcast(v | kamping::v2::views::with_pool(pool));
    EXPECT_EQ(v[0].x, 42);
    EXPECT_DOUBLE_EQ(v[0].y, 3.14);
}
