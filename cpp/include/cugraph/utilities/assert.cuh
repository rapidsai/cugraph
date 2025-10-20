/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>

/**
 * @brief Macro indicating that a location in the code is unreachable.
 *
 * The CUGRAPH_UNREACHABLE macro should only be used where CUGRAPH_FAIL cannot
 * be used due to performance or due to being used in device code. In the
 * majority of host code situations, an exception should be thrown in
 * "unreachable" code paths as those usually aren't tight inner loops like they
 * are in device code.
 *
 * One example where this macro may be used is in conjunction with dispatchers
 * to indicate that a function does not need to return a default value because
 * it has already exhausted all possible cases in a `switch` statement.
 *
 * The assert in this macro can be used when compiling in debug mode to help
 * debug functions that may reach the supposedly unreachable code.
 *
 * Example usage:
 * ```
 * CUGRAPH_UNREACHABLE("Invalid enum value.");
 * ```
 */
#define CUGRAPH_UNREACHABLE(msg)          \
  do {                                    \
    assert(false && "Unreachable: " msg); \
    __builtin_unreachable();              \
  } while (0)
