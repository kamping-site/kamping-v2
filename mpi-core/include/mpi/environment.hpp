// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

/// @file
/// @brief RAII wrapper for MPI initialization and finalization.

#pragma once

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/thread_level.hpp"

namespace mpi::experimental {

/// @brief RAII wrapper that calls \c MPI_Init_thread on construction and \c MPI_Finalize on destruction.
///
/// Construct one instance at the top of \c main. The destructor guards against double-finalization.
/// This class is non-copyable and non-movable: it owns global MPI state.
///
/// @code
/// int main(int argc, char** argv) {
///     mpi::experimental::environment env(argc, argv);                         // single-threaded
///     mpi::experimental::environment env(argc, argv, ThreadLevel::multiple);  // threaded variant
/// }
/// @endcode
class environment {
public:
    /// @brief Initializes MPI without program arguments.
    /// @param thread_level Requested thread support level. Defaults to \c ThreadLevel::single.
    /// @throws mpi_error if \c MPI_Init_thread fails.
    explicit environment(ThreadLevel thread_level = ThreadLevel::single) {
        int provided = 0;
        int err      = MPI_Init_thread(nullptr, nullptr, static_cast<int>(thread_level), &provided);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    }

    /// @brief Initializes MPI with program arguments.
    /// @param argc Argument count (forwarded to \c MPI_Init_thread).
    /// @param argv Argument vector (forwarded to \c MPI_Init_thread).
    /// @param thread_level Requested thread support level. Defaults to \c ThreadLevel::single.
    /// @throws mpi_error if \c MPI_Init_thread fails.
    environment(int& argc, char**& argv, ThreadLevel thread_level = ThreadLevel::single) {
        int provided = 0;
        int err      = MPI_Init_thread(&argc, &argv, static_cast<int>(thread_level), &provided);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    }

    environment(environment const&)            = delete;
    environment& operator=(environment const&) = delete;
    environment(environment&&)                 = delete;
    environment& operator=(environment&&)      = delete;

    /// @brief Calls \c MPI_Finalize if MPI has not been finalized yet.
    ~environment() noexcept {
        int flag = 0;
        MPI_Finalized(&flag);
        if (!flag) {
            MPI_Finalize();
        }
    }

    /// @brief Explicitly finalizes MPI before the destructor runs.
    ///
    /// Use this if you need to handle \c MPI_Finalize errors. The destructor will then skip
    /// the finalization call.
    /// @throws mpi_error if \c MPI_Finalize fails.
    void finalize() {
        int err = MPI_Finalize();
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    }

    /// @brief Returns true if \c MPI_Init has been called.
    [[nodiscard]] static bool initialized() {
        int flag = 0;
        MPI_Initialized(&flag);
        return flag != 0;
    }

    /// @brief Returns true if \c MPI_Finalize has been called.
    [[nodiscard]] static bool finalized() {
        int flag = 0;
        MPI_Finalized(&flag);
        return flag != 0;
    }

    /// @brief Returns the thread support level provided by the MPI implementation after initialization.
    [[nodiscard]] static ThreadLevel thread_level() {
        int provided = 0;
        MPI_Query_thread(&provided);
        return static_cast<ThreadLevel>(provided);
    }
};

} // namespace mpi::experimental
