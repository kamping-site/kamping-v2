// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

/// @file
/// Debugging support, modeled after C++26 `<debugging>`: `is_debugger_present`, `breakpoint` and
/// `breakpoint_if_debugging` for the calling process, and `wait_for_debugger` to stop selected ranks of a
/// communicator until a debugger is attached to them.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <version>

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

#include <mpi.h>

#include "kamping/v2/collectives/barrier.hpp"
#include "kamping/v2/comm.hpp"
#include "kamping/v2/handle.hpp"

// The standard <debugging> is used where the build of dstl has found it to link (DSTL_HAS_STD_DEBUGGING; GCC 16
// declares it but needs libstdc++exp) and the code including this header compiles with it (__cpp_lib_debugging).
#if defined(DSTL_HAS_STD_DEBUGGING) && defined(__cpp_lib_debugging)
    #include <debugging>
    #define DSTL_USE_STD_DEBUGGING 1
#else
    #if defined(__linux__)
        #include <fstream>
    #endif
    #if defined(__APPLE__)
        #include <sys/sysctl.h>
        #include <sys/types.h>
    #endif
#endif
#if __has_include(<unistd.h>)
    #include <unistd.h>
    #define DSTL_DEBUGGING_HAS_UNISTD 1
#endif

namespace dstl {
/// Returns whether a debugger is attached to the calling process, like C++26 `std::is_debugger_present`.
///
/// Calls `std::is_debugger_present` where the standard library provides it. Otherwise it is supported on Linux (a
/// tracer in `/proc/self/status`, which also counts tools like `strace`) and macOS, and returns `false` elsewhere.
inline bool is_debugger_present() noexcept {
#if defined(DSTL_USE_STD_DEBUGGING)
    return std::is_debugger_present();
#elif defined(__linux__)
    try {
        std::ifstream status("/proc/self/status");
        std::string   line;
        while (std::getline(status, line)) {
            std::string_view const key = "TracerPid:";
            if (line.starts_with(key)) {
                auto const first = line.find_first_not_of(" \t", key.size());
                return first != std::string::npos && line[first] != '0';
            }
        }
    } catch (...) {
        // noexcept like std::is_debugger_present: report "no debugger" if the file cannot be read
    }
    return false;
#elif defined(__APPLE__)
    int        mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()};
    kinfo_proc info{};
    size_t     size = sizeof(info);
    if (sysctl(mib, 4, &info, &size, nullptr, 0) != 0) {
        return false;
    }
    return (info.kp_proc.p_flag & P_TRACED) != 0;
#else
    return false;
#endif
}

/// Stops the calling process at this point, like C++26 `std::breakpoint`, which it calls where available.
///
/// Under a debugger, execution stops here and can be continued. Without one, the process is usually terminated by
/// `SIGTRAP`; use `breakpoint_if_debugging` to stop only under a debugger.
inline void breakpoint() noexcept {
#if defined(DSTL_USE_STD_DEBUGGING)
    std::breakpoint();
#elif defined(_MSC_VER)
    __debugbreak();
#elif defined(__i386__) || defined(__x86_64__)
    __asm__ volatile("int3");
#elif defined(__APPLE__) && defined(__aarch64__) && defined(__clang__)
    // lldb on Apple Silicon steps over this trap instruction on continue.
    __builtin_debugtrap();
#elif defined(SIGTRAP)
    // Other trap instructions (e.g. brk on AArch64 Linux) are not stepped over by gdb, so continuing would trap again.
    std::raise(SIGTRAP);
#endif
}

/// Calls `breakpoint` if `is_debugger_present`, like C++26 `std::breakpoint_if_debugging`.
inline void breakpoint_if_debugging() noexcept {
    if (is_debugger_present()) {
        breakpoint();
    }
}

namespace detail {
/// @internal Prints where to attach and blocks the calling process until a debugger is attached.
inline void wait_here_for_debugger(int rank) {
#ifdef DSTL_DEBUGGING_HAS_UNISTD
    char hostname[256] = {};
    gethostname(hostname, sizeof(hostname) - 1);
    std::fprintf(stderr, "dstl: rank %d (PID %d on %s) waiting for a debugger\n", rank, getpid(), hostname);
#else
    std::fprintf(stderr, "dstl: rank %d waiting for a debugger\n", rank);
#endif
    std::fflush(stderr);
    // Where an attached debugger is not detected (is_debugger_present is always false), release the process from the
    // debugger by setting this variable to 0 (gdb: `set var keep_waiting = 0`, lldb: `expr keep_waiting = 0`).
    int volatile keep_waiting = 1;
    while (keep_waiting != 0 && !is_debugger_present()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

/// @internal Whether `rank` is in the list `spec`: ranks and inclusive ranges `a-b` separated by `,`, or `all`. Throws
/// `std::invalid_argument` for a malformed list.
inline bool rank_list_contains(std::string_view spec, int rank) {
    auto const parse_int = [&](std::string_view s) {
        int value            = 0;
        auto const [end, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
        if (s.empty() || ec != std::errc{} || end != s.data() + s.size() || value < 0) {
            throw std::invalid_argument("dstl: malformed rank list \"" + std::string(spec) + "\"");
        }
        return value;
    };
    if (spec == "all") {
        return true;
    }
    bool found = false;
    while (true) {
        auto const       end   = spec.find(',');
        std::string_view token = spec.substr(0, end);
        auto const       dash  = token.find('-');
        int const        first = parse_int(token.substr(0, dash));
        int const        last  = dash == std::string_view::npos ? first : parse_int(token.substr(dash + 1));
        if (first > last) {
            throw std::invalid_argument("dstl: malformed rank list \"" + std::string(spec) + "\"");
        }
        found = found || (first <= rank && rank <= last);
        if (end == std::string_view::npos) {
            return found;
        }
        spec = spec.substr(end + 1);
    }
}
} // namespace detail

/// Stops the given ranks of `comm` until a debugger is attached to each of them.
///
/// Each listed rank prints its PID and host to `stderr` and waits until a debugger is attached; the other ranks wait in
/// a barrier, so all ranks continue together once every listed rank has its debugger. Collective over `comm`.
///
/// A debugger stops the process when it attaches: set breakpoints and continue. On platforms where
/// `is_debugger_present` cannot detect the debugger, set the variable `keep_waiting` in the waiting frame to 0
/// before continuing.
///
/// @param ranks The ranks of `comm` to stop. Ranks that do not exist in `comm` are ignored.
/// @param comm The communicator.
template <kamping::v2::convertible_to_mpi_handle<MPI_Comm> Comm = kamping::v2::comm_view>
void wait_for_debugger(std::span<int const> ranks, Comm const& comm = kamping::v2::comm_world()) {
    kamping::v2::comm_view const cv{kamping::v2::handle(comm)};
    int const                    rank = cv.rank();
    if (std::ranges::find(ranks, rank) != ranks.end()) {
        detail::wait_here_for_debugger(rank);
    }
    kamping::v2::barrier(cv);
}

/// Stops rank `rank` of `comm` until a debugger is attached to it; see `wait_for_debugger(std::span<int const>,
/// Comm const&)`.
template <kamping::v2::convertible_to_mpi_handle<MPI_Comm> Comm = kamping::v2::comm_view>
void wait_for_debugger(int rank, Comm const& comm = kamping::v2::comm_world()) {
    wait_for_debugger(std::span<int const>(&rank, 1), comm);
}

/// Stops the ranks named in an environment variable until a debugger is attached, so that a run can be debugged
/// without recompiling.
///
/// The variable holds ranks of `comm` and inclusive ranges, separated by `,` (e.g. `0,4-7`), or `all`. If it
/// is set, this behaves like `wait_for_debugger` with these ranks and is collective over `comm`; if it is unset or
/// empty, it returns without communicating. Call it right after initializing MPI:
///
/// @code{.cpp}
/// dstl::wait_for_debugger_if_requested();   // mpirun -n 8 -x DSTL_WAIT_FOR_DEBUGGER=0,3 ./app
/// @endcode
///
/// @param comm The communicator.
/// @param env_var The name of the environment variable.
/// @throws std::invalid_argument If the variable does not hold a valid rank list.
template <kamping::v2::convertible_to_mpi_handle<MPI_Comm> Comm = kamping::v2::comm_view>
void wait_for_debugger_if_requested(
    Comm const& comm = kamping::v2::comm_world(), char const* env_var = "DSTL_WAIT_FOR_DEBUGGER"
) {
    char const* const spec = std::getenv(env_var);
    if (spec == nullptr || *spec == '\0') {
        return;
    }
    kamping::v2::comm_view const cv{kamping::v2::handle(comm)};
    int const                    rank = cv.rank();
    if (detail::rank_list_contains(spec, rank)) {
        detail::wait_here_for_debugger(rank);
    }
    kamping::v2::barrier(cv);
}
} // namespace dstl
