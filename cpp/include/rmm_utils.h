/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <sstream>
#include <stdexcept>

#include "utilities/error_utils.h"

#ifndef RMM_TRY
#define RMM_TRY(call)                                                                            \
  do {                                                                                           \
    rmmError_t const status = (call);                                                            \
    if (RMM_SUCCESS != status) { cugraph::detail::throw_rmm_error(status, __FILE__, __LINE__); } \
  } while (0);
#endif

#define RMM_TRY_CUDAERROR(x) \
  if ((x) != RMM_SUCCESS) CUDA_TRY(cudaPeekAtLastError());

#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
