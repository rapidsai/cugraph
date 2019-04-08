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

/* 
 *
 *
 * WARNING: this is a private header file, it should not be publically exposed.
 *
 *
 */

#pragma once
#include "nvgraph.h"
#include "cnmem.h"

#if defined(__cplusplus) 
  extern "C" {
#endif

/* Graph descriptor types */
typedef enum
{
   IS_EMPTY = 0, //nothing
   HAS_TOPOLOGY = 1, //connectivity info
   HAS_VALUES = 2, //MultiValuedCSRGraph
   IS_2D = 3
} nvgraphGraphStatus_t;

struct nvgraphContext {
   cudaStream_t stream;
   cnmemDevice_t cnmem_device;  
   int nvgraphIsInitialized;  
};

struct nvgraphGraphDescr {
   nvgraphGraphStatus_t graphStatus;
   cudaDataType T;							// This is the type of values for the graph
   nvgraphTopologyType_t TT;				// The topology type (class to cast graph_handle pointer to)
   void* graph_handle;						// Opaque pointer to the graph class object
};

#if defined(__cplusplus) 
}//extern "C"
#endif

