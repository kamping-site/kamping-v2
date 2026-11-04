// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstddef>
#include <exception>
#include <optional>
#include <string>

#include <mpi.h>

namespace mpi::experimental {
/// Exception thrown by the MPI wrappers when an MPI call returns an error code other than `MPI_SUCCESS`.
///
/// Errors are only reported this way if the communicator's error handler returns them, e.g. `MPI_ERRORS_RETURN`;
/// with the default `MPI_ERRORS_ARE_FATAL` MPI aborts before the wrapper sees the error.
struct mpi_error : public std::exception {
private:
    int                                _error_code;
    mutable std::optional<std::string> _error_string;

public:
    /// Constructs an error with code `MPI_ERR_UNKNOWN`.
    mpi_error() : mpi_error(MPI_ERR_UNKNOWN) {}

    /// Constructs an error from the error code returned by an MPI call.
    explicit mpi_error(int error_code) : _error_code(error_code) {}

    /// @return The MPI error code.
    [[nodiscard]] int error_code() const {
        return _error_code;
    }

    /// @return The MPI error class of the error code (`MPI_Error_class`).
    [[nodiscard]] int error_class() const {
        int error_class = 0;
        int err         = MPI_Error_class(_error_code, &error_class);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        return error_class;
    }

    /// @return An error whose code is the error class of this error.
    [[nodiscard]] mpi_error as_class_error() const {
        return mpi_error(this->error_class());
    }

    /// @return The description of the error code (`MPI_Error_string`).
    [[nodiscard]] std::string error_string() const {
        std::string error_string;
        error_string.resize(MPI_MAX_ERROR_STRING);
        int error_string_len = 0;
        int err              = MPI_Error_string(_error_code, error_string.data(), &error_string_len);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        error_string.resize(static_cast<std::size_t>(error_string_len));
        return error_string;
    }

    /// @return The description of the error code, computed on first use.
    [[nodiscard]] char const* what() const noexcept final {
        if (!_error_string) {
            try {
                _error_string = error_string();
            } catch (...) {
                _error_string = "<failed to get error string>";
            }
        }
        return _error_string->data();
    }
};
} // namespace mpi::experimental
