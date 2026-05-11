/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Symbol visibility macros for cuGraph shared libraries.
// When CXX_VISIBILITY_PRESET is set to hidden, only symbols explicitly
// marked with CUGRAPH_EXPORT will be visible in the shared library.
// CUGRAPH_HIDDEN can be used to explicitly mark symbols as hidden.
#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))
#define CUGRAPH_EXPORT __attribute__((visibility("default")))
#define CUGRAPH_HIDDEN __attribute__((visibility("hidden")))
#else
#define CUGRAPH_EXPORT
#define CUGRAPH_HIDDEN
#endif
