// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

// Calls every overload of the kamping::v2 operations, checking the values used for omitted arguments: send tag
// DEFAULT_SEND_TAG, any source and any tag for receives, root 0, std::plus<> as reduction operation.

#include <functional>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2.hpp"

namespace v2 = kamping::v2;

namespace {
v2::comm_view world() {
    return v2::comm_view{MPI_COMM_WORLD};
}
} // namespace

TEST(OverloadsTest, SendRecv) {
    auto comm = world();
    if (comm.size() < 2 || comm.rank() > 1) {
        return;
    }
    if (comm.rank() == 0) {
        v2::send(std::vector<int>{1}, 1, comm);                             // tag 0
        v2::send(std::vector<int>{2}, 1, 7, comm);                          // tag 7
        v2::send(v2::send_mode::sync, std::vector<int>{3}, 1, comm);        // tag 0
        v2::send(v2::send_mode::standard, std::vector<int>{4}, 1, 8, comm); // tag 8
    } else {
        std::vector<int> buf(1);
        MPI_Status       status;
        v2::recv(buf, 0, v2::DEFAULT_SEND_TAG, comm);
        EXPECT_EQ(buf[0], 1);
        v2::recv(buf, 0, comm, &status); // any tag
        EXPECT_EQ(buf[0], 2);
        EXPECT_EQ(status.MPI_TAG, 7);
        v2::recv(buf, comm, &status); // any source, any tag
        EXPECT_EQ(buf[0], 3);
        EXPECT_EQ(status.MPI_SOURCE, 0);
        EXPECT_EQ(status.MPI_TAG, v2::DEFAULT_SEND_TAG);
        auto received = v2::recv(std::vector<int>{} | v2::views::resize, 0, 8, comm);
        EXPECT_EQ(received.underlying(), std::vector<int>{4});
    }
}

TEST(OverloadsTest, NonBlockingSendRecv) {
    auto comm = world();
    if (comm.size() < 2 || comm.rank() > 1) {
        return;
    }
    if (comm.rank() == 0) {
        std::vector<int> a{1}, b{2}, c{3}, d{4};
        auto             r1 = v2::isend(a, 1, comm);
        auto             r2 = v2::isend(b, 1, 5, comm);
        v2::request      r3;
        v2::isend(r3, c, 1, comm);
        v2::request r4;
        v2::isend(v2::send_mode::sync, r4, d, 1, 6, comm);
        r1.wait();
        r2.wait();
        r3.wait();
        r4.wait();
    } else {
        std::vector<int> buf(1);
        EXPECT_EQ(v2::irecv(std::vector<int>(1), 0, v2::DEFAULT_SEND_TAG, comm).wait(), std::vector<int>{1});
        EXPECT_EQ(v2::irecv(std::vector<int>(1), 0, comm).wait(), std::vector<int>{2});
        v2::request req;
        v2::irecv(req, buf, comm);
        req.wait();
        EXPECT_EQ(buf[0], 3);
        v2::irecv(req, buf, 0, 6, comm);
        req.wait();
        EXPECT_EQ(buf[0], 4);
    }
}

TEST(OverloadsTest, SendrecvUntaggedReceivesAnyTag) {
    auto comm = world();
    if (comm.size() < 2 || comm.rank() > 1) {
        return;
    }
    int const peer = 1 - comm.rank();
    if (comm.rank() == 0) {
        // Sends with tag 3; the peer's untagged sendrecv must still receive it.
        auto [_, r] = v2::sendrecv(std::vector<int>{10}, peer, 3, std::vector<int>(1), peer, MPI_ANY_TAG, comm);
        EXPECT_EQ(r, std::vector<int>{11});
    } else {
        MPI_Status status;
        auto [_, r] = v2::sendrecv(std::vector<int>{11}, peer, std::vector<int>(1), comm, &status);
        EXPECT_EQ(r, std::vector<int>{10});
        EXPECT_EQ(status.MPI_TAG, 3);
    }
}

TEST(OverloadsTest, Probe) {
    auto comm = world();
    if (comm.size() < 2 || comm.rank() > 1) {
        return;
    }
    if (comm.rank() == 0) {
        for (int tag = 0; tag < 3; ++tag) {
            v2::send(std::vector<int>{tag}, 1, tag, comm);
        }
    } else {
        MPI_Status status;
        v2::probe(comm, &status);
        EXPECT_EQ(status.MPI_TAG, 0);
        auto m0 = v2::mprobe(0, 0, comm);
        v2::mrecv(std::vector<int>{} | v2::views::resize, std::move(m0));
        while (!v2::iprobe(0, comm)) {
        }
        auto m1 = v2::mprobe(0, comm);
        EXPECT_EQ(m1.tag(), 1);
        v2::mrecv(std::vector<int>{} | v2::views::resize, std::move(m1));
        auto m2 = v2::mprobe(comm);
        EXPECT_EQ(m2.tag(), 2);
        v2::mrecv(std::vector<int>{} | v2::views::resize, std::move(m2));
        EXPECT_FALSE(v2::improbe(comm).has_value());
    }
}

TEST(OverloadsTest, RootedCollectives) {
    auto      comm = world();
    int const size = comm.size();
    int const last = size - 1;
    bool      on_0 = comm.rank() == 0;

    // bcast: from rank 0 by default
    std::vector<int> v = on_0 ? std::vector<int>{42} : std::vector<int>{};
    v2::bcast(v | v2::views::resize, comm);
    EXPECT_EQ(v, std::vector<int>{42});
    std::vector<int> w{comm.rank()};
    v2::bcast(w, last, comm);
    EXPECT_EQ(w, std::vector<int>{last});

    // gather: two-buffer form on root, shorthand elsewhere
    std::vector<int> gathered;
    if (on_0) {
        v2::gather(std::vector<int>{comm.rank()}, gathered | v2::views::resize, comm);
        ASSERT_EQ(static_cast<int>(gathered.size()), size);
        EXPECT_EQ(gathered.back(), last);
    } else {
        v2::gather(std::vector<int>{comm.rank()}, comm);
    }
    if (comm.rank() == last) {
        auto [_, r] = v2::gather(std::vector<int>{1}, std::vector<int>{} | v2::views::resize, last, comm);
        EXPECT_EQ(static_cast<int>(r.underlying().size()), size);
    } else {
        v2::gather(std::vector<int>{1}, last, comm);
    }

    // scatter: one element per rank
    std::vector<int> all(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i) {
        all[static_cast<std::size_t>(i)] = 10 * i;
    }
    std::vector<int> mine(1);
    if (on_0) {
        v2::scatter(all, mine, comm);
    } else {
        v2::scatter(mine, comm);
    }
    EXPECT_EQ(mine[0], 10 * comm.rank());
    if (comm.rank() == last) {
        v2::scatter(all, mine, last, comm);
    } else {
        v2::scatter(mine, last, comm);
    }
    EXPECT_EQ(mine[0], 10 * comm.rank());

    // scatterv: one element per rank
    std::vector<int> counts(static_cast<std::size_t>(size), 1);
    if (on_0) {
        v2::scatterv(all | v2::views::with_counts(counts) | v2::views::auto_displs(), mine, comm);
    } else {
        v2::scatterv(mine, comm);
    }
    EXPECT_EQ(mine[0], 10 * comm.rank());

    // gatherv: explicit counts on root
    if (on_0) {
        std::vector<int> recv(static_cast<std::size_t>(size));
        v2::gatherv(
            std::vector<int>{comm.rank()},
            recv | v2::views::with_counts(counts) | v2::views::auto_displs(),
            comm
        );
        EXPECT_EQ(recv.back(), last);
    } else {
        v2::gatherv(std::vector<int>{comm.rank()}, v2::null_buf_v, comm);
    }
}

TEST(OverloadsTest, Reductions) {
    auto      comm     = world();
    int const size     = comm.size();
    int const last     = size - 1;
    int const expected = size * (size - 1) / 2;
    int       rank     = comm.rank();
    int       result   = 0;

    // reduce: sum to rank 0 by default
    if (rank == 0) {
        v2::reduce(v2::views::ref_single(rank), v2::views::ref_single(result), comm);
        EXPECT_EQ(result, expected);
        v2::reduce(v2::views::ref_single(rank), v2::views::ref_single(result), std::multiplies<>{}, comm);
        EXPECT_EQ(result, 0);
    } else {
        v2::reduce(v2::views::ref_single(rank), comm);
        v2::reduce(v2::views::ref_single(rank), std::multiplies<>{}, comm);
    }
    if (rank == last) {
        v2::reduce(v2::views::ref_single(rank), v2::views::ref_single(result), MPI_MAX, last, comm);
        EXPECT_EQ(result, last);
    } else {
        v2::reduce(v2::views::ref_single(rank), MPI_MAX, last, comm);
    }

    v2::allreduce(v2::views::ref_single(rank), v2::views::ref_single(result), comm);
    EXPECT_EQ(result, expected);
    v2::allreduce(v2::views::ref_single(rank), v2::views::ref_single(result), MPI_MAX, comm);
    EXPECT_EQ(result, last);
    v2::scan(v2::views::ref_single(rank), v2::views::ref_single(result), comm);
    EXPECT_EQ(result, rank * (rank + 1) / 2);
    v2::exscan(v2::views::ref_single(rank), v2::views::ref_single(result), std::plus<>{}, comm);
    if (rank > 0) {
        EXPECT_EQ(result, rank * (rank - 1) / 2);
    }
}

TEST(OverloadsTest, Barriers) {
    auto comm = world();
    v2::barrier(comm);
    v2::ibarrier(comm).wait();
}
