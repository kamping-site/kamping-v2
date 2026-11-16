// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

/// @file
/// @brief RAII wrapper for MPI initialization and finalization.

#pragma once

#include <mpi.h>

#include "kamping/v2/kassert.hpp"
#include "mpi/error.hpp"
#include "mpi/thread_level.hpp"

namespace mpi::experimental {

/// @brief RAII wrapper for the World Model: calls \c MPI_Init_thread on construction and \c MPI_Finalize on
/// destruction.
///
/// Construct one instance at the top of \c main. The destructor guards against double-finalization.
/// This class is non-copyable and non-movable: it owns global MPI state. Its static members query the World Model and
/// work no matter who initialized MPI: an `environment`, a test main or another library.
///
/// @code
/// int main(int argc, char** argv) {
///     mpi::experimental::environment env(argc, argv);                         // single-threaded
///     mpi::experimental::environment env(argc, argv, thread_level::multiple);  // threaded variant
/// }
/// @endcode
class environment {
public:
    /// @brief Initializes MPI without program arguments.
    /// @param requested Requested thread support level. Defaults to \c thread_level::single. MPI may provide a
    /// lower one; see provided_thread_level().
    /// @throws mpi_error if \c MPI_Init_thread fails.
    explicit environment(thread_level requested = thread_level::single) {
        int provided = 0;
        int err      = MPI_Init_thread(nullptr, nullptr, static_cast<int>(requested), &provided);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    }

    /// @brief Initializes MPI with program arguments.
    /// @param argc Argument count (forwarded to \c MPI_Init_thread).
    /// @param argv Argument vector (forwarded to \c MPI_Init_thread).
    /// @param requested Requested thread support level. Defaults to \c thread_level::single. MPI may provide a
    /// lower one; see provided_thread_level().
    /// @throws mpi_error if \c MPI_Init_thread fails.
    environment(int& argc, char**& argv, thread_level requested = thread_level::single) {
        int provided = 0;
        int err      = MPI_Init_thread(&argc, &argv, static_cast<int>(requested), &provided);
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

    /// @brief The thread support level provided for the World Model (\c MPI_Query_thread), i.e., the level granted by
    /// \c MPI_Init_thread, which may be lower than the requested one.
    ///
    /// It only applies to the World Model: objects derived from a session have the level of that session (see
    /// `session::provided_thread_level()`). Code that is only given a communicator cannot tell which one applies, so
    /// library code should state the level it needs as a precondition instead of checking this.
    /// @pre MPI is initialized and not yet finalized.
    /// @throws mpi_error if \c MPI_Query_thread fails.
    [[nodiscard]] static thread_level provided_thread_level() {
        KAMPING_V2_ASSERT(
            initialized() && !finalized(),
            "environment::provided_thread_level() requires MPI to be initialized and not yet finalized"
        );
        int provided = 0;
        int err      = MPI_Query_thread(&provided);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        return static_cast<thread_level>(provided);
    }
};

} // namespace mpi::experimental
