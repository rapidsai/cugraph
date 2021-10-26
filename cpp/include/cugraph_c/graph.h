/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cugraph_c/cugraph_api.h>
#include <cugraph_c/array.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int align_;
} cugraph_graph_t;

/* SG graph */
/**
 * @brief     Construct an SG graph
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  src         Device array containing the source vertex ids
 * @param [in]  dst         Device array containing the destination vertex ids
 * @param [in]  weights     Device array containing the edge weights
 * @param [in]  store_transposed If true create the graph initially in transposed format
 * @param [in]  renumber    If true, renumber vertices to make an efficient data structure.
 *    If false, do not renumber.  Renumbering is required if the vertices are not sequential
 *    integer values from 0 to num_vertices.
 * @param [in]  do_expensive_check    If true, do expensive checks to validate the input data
 *    is consistent with software assumptions.  If false bypass these checks.
 * @param [in]  is_symmetric If true the input graph is symmetric
 * @param [in]  is_multigraph If true the input graph is a multi graph (can have multiple edges
 *    between a pair of vertices)
 * @param [out] graph       A pointer to the graph object
 */
cugraph_error_t cugraph_sg_graph_create(const cugraph_raft_handle_t* handle,
                                        const cugraph_type_erased_device_array_t* src,
                                        const cugraph_type_erased_device_array_t* dst,
                                        const cugraph_type_erased_device_array_t* weights,
                                        bool_t store_transposed,
                                        bool_t renumber,
                                        bool_t check,
                                        bool_t is_symmetric,
                                        bool_t is_multigraph,
                                        cugraph_graph_t **graph);

void cugraph_sg_graph_free(cugraph_graph_t* graph);

/* MG graph */
cugraph_graph_t* cugraph_mg_graph_create(
  const cugraph_raft_handle_t* handle,
  const cugraph_type_erased_device_array_t* src,
  const cugraph_type_erased_device_array_t* dst,
  const cugraph_type_erased_device_array_t* weights,
  const cugraph_type_erased_host_array_t* vertex_partition_offsets,
  const cugraph_type_erased_host_array_t* segment_offsets,
  size_t num_segments,
  bool_t store_transposed,
  size_t num_vertices,
  size_t num_edges,
  bool_t check,
  bool_t is_symmetric,
  bool_t is_multigraph);

void cugraph_mg_graph_free(cugraph_graph_t* graph);

#ifdef __cplusplus
}
#endif
