/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/random.h>
#include <cugraph_c/resource_handle.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief       Opaque COO definition
 */
typedef struct {
  int32_t align_;
} cugraph_coo_t;

/**
 * @brief       Opaque COO list definition
 */
typedef struct {
  int32_t align_;
} cugraph_coo_list_t;

/**
 * @brief       Get the source vertex ids
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of source vertex ids
 */
cugraph_type_erased_device_array_view_t* cugraph_coo_get_sources(cugraph_coo_t* coo);

/**
 * @brief       Get the destination vertex ids
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of destination vertex ids
 */
cugraph_type_erased_device_array_view_t* cugraph_coo_get_destinations(cugraph_coo_t* coo);

/**
 * @brief       Get the edge weights
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of edge weights, NULL if no edge weights in COO
 */
cugraph_type_erased_device_array_view_t* cugraph_coo_get_edge_weights(cugraph_coo_t* coo);

/**
 * @brief       Get the edge id
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of edge id, NULL if no edge ids in COO
 */
cugraph_type_erased_device_array_view_t* cugraph_coo_get_edge_id(cugraph_coo_t* coo);

/**
 * @brief       Get the edge type
 *
 * @param [in]     coo   Opaque pointer to COO
 * @return type erased array view of edge type, NULL if no edge types in COO
 */
cugraph_type_erased_device_array_view_t* cugraph_coo_get_edge_type(cugraph_coo_t* coo);

/**
 * @brief       Get the number of coo object in the list
 *
 * @param [in]     coo_list   Opaque pointer to COO list
 * @return number of elements
 */
size_t cugraph_coo_list_size(const cugraph_coo_list_t* coo_list);

/**
 * @brief       Get a COO from the list
 *
 * @param [in]     coo_list   Opaque pointer to COO list
 * @param [in]     index      Index of desired COO from list
 * @return a cugraph_coo_t* object from the list
 */
cugraph_coo_t* cugraph_coo_list_element(cugraph_coo_list_t* coo_list, size_t index);

/**
 * @brief     Free coo object
 *
 * @param [in]    coo Opaque pointer to COO
 */
void cugraph_coo_free(cugraph_coo_t* coo);

/**
 * @brief     Free coo list
 *
 * @param [in]    coo_list Opaque pointer to list of COO objects
 */
void cugraph_coo_list_free(cugraph_coo_list_t* coo_list);

#ifdef __cplusplus
}
#endif
