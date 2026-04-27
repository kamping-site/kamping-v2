#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/p2p/recv.hpp"
#include "kamping/v2/p2p/send.hpp"
#include "kamping/v2/views.hpp"

// ── PMPI intercept ────────────────────────────────────────────────────────────

static int g_probe_calls  = 0;
static int g_mprobe_calls = 0;

extern "C" {
int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status* status) {
    ++g_probe_calls;
    return PMPI_Probe(source, tag, comm, status);
}
int MPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message* message, MPI_Status* status) {
    ++g_mprobe_calls;
    return PMPI_Mprobe(source, tag, comm, message, status);
}
}

// ── Opt-out buffer type ───────────────────────────────────────────────────────

struct no_probe_buf {
    std::vector<int> data;

    void          set_recv_count(std::ptrdiff_t n) { data.resize(static_cast<std::size_t>(n)); }
    std::ptrdiff_t mpi_count() const { return static_cast<std::ptrdiff_t>(data.size()); }
    int*          mpi_ptr() { return data.data(); }
    MPI_Datatype  mpi_type() const { return MPI_INT; }
};

template <>
inline constexpr bool kamping::v2::use_matched_probe<no_probe_buf> = false;

// ── Compile-time trait checks ─────────────────────────────────────────────────

static_assert(kamping::v2::use_matched_probe<std::vector<int>>);
static_assert(!kamping::v2::use_matched_probe<no_probe_buf>);
static_assert(!kamping::v2::use_matched_probe<
              kamping::v2::resize_view<kamping::v2::ref_view<no_probe_buf>>>);

// ── MPI functional tests ──────────────────────────────────────────────────────

class V2RecvTest : public ::testing::Test {
protected:
    void SetUp() override {
        g_probe_calls  = 0;
        g_mprobe_calls = 0;
    }
};

// Default path: resize view on a plain vector must use MPI_Mprobe + MPI_Mrecv.
TEST_F(V2RecvTest, MatchedProbeUsedForResizableVector) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) GTEST_SKIP();

    if (rank == 0) {
        std::vector<int> data = {10, 20, 30};
        kamping::v2::send(data, 1, 0);
    } else if (rank == 1) {
        std::vector<int> buf;
        kamping::v2::recv(buf | kamping::v2::views::resize, 0, 0);
        EXPECT_EQ(buf, (std::vector<int>{10, 20, 30}));
        EXPECT_EQ(g_mprobe_calls, 1);
        EXPECT_EQ(g_probe_calls, 0);
    }
}

// Opt-out path: no_probe_buf must use MPI_Probe + MPI_Recv, not MPI_Mprobe.
TEST_F(V2RecvTest, UnmatchedProbeUsedForOptOutBuffer) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) GTEST_SKIP();

    if (rank == 0) {
        std::vector<int> data = {10, 20, 30};
        kamping::v2::send(data, 1, 0);
    } else if (rank == 1) {
        no_probe_buf buf;
        kamping::v2::recv(buf, 0, 0);
        EXPECT_EQ(buf.data, (std::vector<int>{10, 20, 30}));
        EXPECT_EQ(g_probe_calls, 1);
        EXPECT_EQ(g_mprobe_calls, 0);
    }
}

// Wildcard receive with opt-out buffer: verifies the resolved source/tag from
// MPI_Probe are forwarded to MPI_Recv so the right message is consumed.
TEST_F(V2RecvTest, UnmatchedProbeWildcardSourceResolvesCorrectly) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) GTEST_SKIP();

    if (rank == 0) {
        std::vector<int> data = {rank};
        kamping::v2::send(data, 1, 42);
    } else if (rank == 1) {
        no_probe_buf buf;
        kamping::v2::recv(buf, MPI_ANY_SOURCE, 42);
        EXPECT_EQ(buf.data, (std::vector<int>{0}));
        EXPECT_EQ(g_probe_calls, 1);
        EXPECT_EQ(g_mprobe_calls, 0);
    }
}
