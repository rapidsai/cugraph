/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cugraph_c/error.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bool_ { FALSE = 0, TRUE = 1 } bool_t;

typedef int8_t byte_t;

typedef enum data_type_id_ { INT32 = 0, INT64, FLOAT32, FLOAT64, SIZE_T, NTYPES } data_type_id_t;

typedef struct cugraph_resource_handle_ {
  int32_t align_;
} cugraph_resource_handle_t;

// FIXME: Don't really want a raft handle here.  We really want to be able to
//        configure the resource handle ourselves.  But that requires a bunch
//        of logic that's currently only available in python.
/**
 * @brief     Construct a resource handle
 *
 * @param [in]  raft_handle   Handle for accessing resources
 *                            If NULL, we will create a raft handle
 *                            internally
 *
 * @return A graph resource handle
 */
cugraph_resource_handle_t* cugraph_create_resource_handle(void* raft_handle);

/**
 * @brief get rank from resource handle
 *
 * If the resource handle has been configured for multi-gpu, this will return
 * the rank for this worker.  If the resource handle has not been configured for
 * multi-gpu this will always return 0.
 *
 * @param [in]  handle          Handle for accessing resources
 * @return rank
 */
int cugraph_resource_handle_get_rank(const cugraph_resource_handle_t* handle);

/**
 * @brief     Free resources in the resource handle
 *
 * @param [in]  handle          Handle for accessing resources
 */
void cugraph_free_resource_handle(cugraph_resource_handle_t* handle);

#ifdef __cplusplus
}
#endif
