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

#define RMM_TRY_THROW( call )  if ((call)!=RMM_SUCCESS) \
    {                                                   \
      std::stringstream ss;                             \
      ss << "ERROR: RMM runtime call  " << #call        \
         << cudaGetErrorString(cudaGetLastError());     \
      throw std::runtime_error(ss.str());               \
    }

#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#define ALLOC_TRY( ptr, sz, stream ){                   \
      RMM_TRY_THROW( RMM_ALLOC((ptr), (sz), (stream)) ) \
    }

//TODO: change this when RMM alloc managed will be available !!!!!
#define ALLOC_MANAGED_TRY(ptr, sz, stream){         \
  RMM_TRY_THROW( RMM_ALLOC((ptr), (sz), (stream)) ) \
}

#define REALLOC_TRY(ptr, new_sz, stream){             \
  RMM_TRY_THROW( RMM_REALLOC((ptr), (sz), (stream)) ) \
}

#define ALLOC_FREE_TRY(ptr, stream){                \
  RMM_TRY_THROW( RMM_FREE( (ptr), (stream) ) )  \
}
