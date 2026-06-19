// Copyright (c) 2026 Karlsruhe Institute of Technology
// SPDX-License-Identifier: BSL-1.0

#pragma once

/// @file
/// Shared entry point declaration for the dstl tests that need MPI_THREAD_MULTIPLE. The stock
/// KaTestrophe main calls plain MPI_Init, which does not request an elevated thread level, so
/// thread_multiple_test_main.cpp provides a custom main that calls MPI_Init_thread(MPI_THREAD_MULTIPLE)
/// and records the level the runtime actually provided so individual tests can GTEST_SKIP when it is
/// insufficient.

/// @return The thread support level the MPI runtime provided at initialization (an MPI_THREAD_* value).
int provided_thread_level();
