// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/ibarrier.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/completion.hpp"
#include "kamping/v2/request.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "mpi/collectives/ibarrier.hpp"
#include "mpi/comm.hpp"
#include "mpi/completion/wait.hpp"
#include "mpi/p2p/recv.hpp"
#include "mpi/p2p/send.hpp"
#include "mpi/request.hpp"

TEST(IBarrierTest, core_ibarrier) {
    {
        MPI_Request req = MPI_REQUEST_NULL;
        mpi::experimental::ibarrier(MPI_COMM_WORLD, req);
        kamping::v2::wait(req);
    }
    {
        kamping::v2::request   req;
        kamping::v2::comm_view world{MPI_COMM_WORLD};
        mpi::experimental::ibarrier(world, req);
        req.wait();
    }
}

TEST(IBarrierTest, v2_ibarrier) {
    auto req = kamping::v2::ibarrier(MPI_COMM_WORLD);
    req.wait();
    EXPECT_TRUE(req.is_null());
}

TEST(IBarrierTest, ibarrier_does_not_complete_before_all_ranks_entered) {
    // The ibarrier must be the only collective on `world` here: all ranks have to call collectives on a
    // communicator in the same order, so rank 0 releases the others with point-to-point messages instead.
    kamping::v2::request   req;
    kamping::v2::comm_view world{MPI_COMM_WORLD};
    if (world.rank() == 0) {
        mpi::experimental::ibarrier(world, req);
        if (world.size() > 1) {
            // no other rank has entered the barrier yet
            EXPECT_FALSE(req.test());
        }
        for (int r = 1; r < world.size(); ++r) {
            int token = 0;
            mpi::experimental::send(kamping::v2::views::ref_single(token), r, 0, MPI_COMM_WORLD);
        }
    } else {
        int token = 0;
        mpi::experimental::recv(kamping::v2::views::ref_single(token), 0, MPI_ANY_TAG, MPI_COMM_WORLD);
        mpi::experimental::ibarrier(world, req);
    }
    req.wait();
}
