// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

// Each scenario drops a still-active request and must terminate with a diagnostic. Every scenario runs as its own
// process (see CMakeLists.txt): the test passes if the diagnostic is printed, and fails if "NOT TERMINATED" is
// printed.
//
// The active request is always an irecv from self without a matching send, so it can never complete.

#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "kamping/v2/iresult.hpp"
#include "mpi/p2p/irecv.hpp"
#include "mpi/request.hpp"
#include "mpi/request_vector.hpp"

namespace me = mpi::experimental;

namespace {

int self() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int recv_buffer = 0;

void post_unmatched_irecv(MPI_Request* request) {
    me::irecv(std::span{&recv_buffer, 1}, self(), 0, MPI_COMM_WORLD, request);
}

void request_destructor() {
    me::request r;
    post_unmatched_irecv(r.mpi_handle_ptr());
}

void request_move_assignment() {
    me::request r;
    post_unmatched_irecv(r.mpi_handle_ptr());
    r = me::request{};
}

void request_vector_destructor() {
    me::request_vector requests;
    requests.emplace_back();                                        // null slots are fine ...
    post_unmatched_irecv(requests.emplace_back().mpi_handle_ptr()); // ... active ones are not
}

void request_vector_clear() {
    me::request_vector requests;
    post_unmatched_irecv(requests.emplace_back().mpi_handle_ptr());
    requests.clear();
}

void request_vector_resize() {
    me::request_vector requests;
    post_unmatched_irecv(requests.emplace_back().mpi_handle_ptr());
    requests.emplace_back();
    requests.resize(1); // dropping the null slot is fine ...
    requests.resize(0); // ... dropping the active one is not
}

void request_vector_pop_back() {
    me::request_vector requests;
    post_unmatched_irecv(requests.emplace_back().mpi_handle_ptr());
    requests.pop_back();
}

void iresult_destructor() {
    kamping::v2::iresult<std::vector<int>> res(std::vector<int>(1));
    me::irecv(res.view(), self(), 0, MPI_COMM_WORLD, res.mpi_handle_ptr());
}

void iresult_during_unwinding() {
    try {
        kamping::v2::iresult<std::vector<int>> res(std::vector<int>(1));
        me::irecv(res.view(), self(), 0, MPI_COMM_WORLD, res.mpi_handle_ptr());
        throw std::runtime_error("unwinding");
    } catch (std::runtime_error const&) {
    }
}

/// Control: completed or released requests must not terminate.
void completed_requests() {
    me::request_vector requests;
    requests.emplace_back();
    me::request r;
    post_unmatched_irecv(r.mpi_handle_ptr());
    MPI_Request raw = r.release();
    MPI_Cancel(&raw);
    MPI_Wait(&raw, MPI_STATUS_IGNORE);
    std::puts("CONTROL OK");
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    if (argc != 2) {
        std::fputs("usage: terminate_on_active_request_test <scenario>\n", stderr);
        return 2;
    }
    char const* scenario = argv[1];
    if (std::strcmp(scenario, "request_destructor") == 0) {
        request_destructor();
    } else if (std::strcmp(scenario, "request_move_assignment") == 0) {
        request_move_assignment();
    } else if (std::strcmp(scenario, "request_vector_destructor") == 0) {
        request_vector_destructor();
    } else if (std::strcmp(scenario, "request_vector_clear") == 0) {
        request_vector_clear();
    } else if (std::strcmp(scenario, "request_vector_resize") == 0) {
        request_vector_resize();
    } else if (std::strcmp(scenario, "request_vector_pop_back") == 0) {
        request_vector_pop_back();
    } else if (std::strcmp(scenario, "iresult_destructor") == 0) {
        iresult_destructor();
    } else if (std::strcmp(scenario, "iresult_during_unwinding") == 0) {
        iresult_during_unwinding();
    } else if (std::strcmp(scenario, "completed_requests") == 0) {
        completed_requests();
        MPI_Finalize();
        return 0;
    } else {
        std::fprintf(stderr, "unknown scenario %s\n", scenario);
        return 2;
    }
    std::puts("NOT TERMINATED");
    MPI_Finalize();
    return 0;
}
