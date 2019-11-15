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

#define RMM_TRY(call)                                             \
  do {                                                            \
    rmmError_t const status = (call);                             \
    if (RMM_SUCCESS != status) {                                  \
      cugraph::detail::throw_rmm_error(status, __FILE__, __LINE__);  \
    }                                                             \
  } while (0);

#define RMM_TRY_CUDAERROR(x) \
  if ((x) != RMM_SUCCESS) CUDA_TRY(cudaPeekAtLastError());
  
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#define ALLOC_TRY( ptr, sz, stream ){               \
  RMM_TRY( RMM_ALLOC((ptr), (sz), (stream)) ) \
}

#define REALLOC_TRY(ptr, new_sz, stream){             \
  RMM_TRY( RMM_REALLOC((ptr), (sz), (stream)) ) \
}

// TODO: temporarily wrapping RMM_FREE in a rmmIsInitialized() check to work
// around the RMM session being finalized prior to this call. A larger
// refactoring will need to be done to eliminate the need to do this, and
// calling RMM APIs directly should likely also be removed in favor of working
// with a higher-level abstraction that manages RMM properly (eg. cuDF?)
#define ALLOC_FREE_TRY(ptr, stream){              \
  if(rmmIsInitialized((rmmOptions_t*) NULL)) {    \
    RMM_TRY( RMM_FREE( (ptr), (stream) ) )  \
  }                                               \
}
