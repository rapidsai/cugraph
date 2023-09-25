/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

typedef enum { POWER_LAW = 0, UNIFORM } cugraph_generator_distribution_t;

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

/**
 * @brief      Generate RMAT edge list
 *
 * Returns a COO containing edges generated from the RMAT generator.
 *
 * Vertex types will be int32 if scale < 32 and int64 if scale >= 32
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [in/out] rng_state          State of the random number generator, updated with each call
 * @param [in]     scale Scale factor to set the number of vertices in the graph. Vertex IDs have
 * values in [0, V), where V = 1 << @p scale.
 * @param [in]     num_edges          Number of edges to generate.
 * @param [in]     a                  a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator
 * (vist https://graph500.org for additional details). a, b, c, d should be non-negative and a + b +
 * c should be no larger than 1.0.
 * @param [in]     b                  a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator
 * (vist https://graph500.org for additional details). a, b, c, d should be non-negative and a + b +
 * c should be no larger than 1.0.
 * @param [in]     c                  a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator
 * (vist https://graph500.org for additional details). a, b, c, d should be non-negative and a + b +
 * c should be no larger than 1.0.
 * @param [in]     clip_and_flip      Flag controlling whether to generate edges only in the lower
 * triangular part (including the diagonal) of the graph adjacency matrix (if set to `true`) or not
 * (if set to `false`).
 * @param [in]     scramble_vertex_ids Flag controlling whether to scramble vertex ID bits
 * (if set to `true`) or not (if set to `false`); scrambling vertex ID bits breaks correlation
 * between vertex ID values and vertex degrees.
 * @param [out]    result             Opaque pointer to generated coo
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_generate_rmat_edgelist(const cugraph_resource_handle_t* handle,
                                                    cugraph_rng_state_t* rng_state,
                                                    size_t scale,
                                                    size_t num_edges,
                                                    double a,
                                                    double b,
                                                    double c,
                                                    bool_t clip_and_flip,
                                                    bool_t scramble_vertex_ids,
                                                    cugraph_coo_t** result,
                                                    cugraph_error_t** error);

/**
 * @brief      Generate RMAT edge lists
 *
 * Returns a COO list containing edges generated from the RMAT generator.
 *
 * Vertex types will be int32 if scale < 32 and int64 if scale >= 32
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [in/out] rng_state          State of the random number generator, updated with each call
 * @param [in]     n_edgelists Number of edge lists (graphs) to generate
 * @param [in]     min_scale Scale factor to set the minimum number of verties in the graph.
 * @param [in]     max_scale Scale factor to set the maximum number of verties in the graph.
 * @param [in]     edge_factor Average number of edges per vertex to generate.
 * @param [in]     size_distribution Distribution of the graph sizes, impacts the scale parameter of
 * the R-MAT generator
 * @param [in]     edge_distribution Edges distribution for each graph, impacts how R-MAT parameters
 * a,b,c,d, are set.
 * @param [in]     clip_and_flip      Flag controlling whether to generate edges only in the lower
 * triangular part (including the diagonal) of the graph adjacency matrix (if set to `true`) or not
 * (if set to `false`).
 * @param [in]     scramble_vertex_ids Flag controlling whether to scramble vertex ID bits
 * (if set to `true`) or not (if set to `false`); scrambling vertex ID bits breaks correlation
 * between vertex ID values and vertex degrees.
 * @param [out]    result             Opaque pointer to generated coo list
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_generate_rmat_edgelists(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  size_t n_edgelists,
  size_t min_scale,
  size_t max_scale,
  size_t edge_factor,
  cugraph_generator_distribution_t size_distribution,
  cugraph_generator_distribution_t edge_distribution,
  bool_t clip_and_flip,
  bool_t scramble_vertex_ids,
  cugraph_coo_list_t** result,
  cugraph_error_t** error);

/**
 * @brief      Generate edge weights and add to an rmat edge list
 *
 * Updates a COO to contain random edge weights
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [in/out] rng_state          State of the random number generator, updated with each call
 * @param [in/out] coo                Opaque pointer to the coo, weights will be added (overwriting
 * any existing weights)
 * @param [in]     dtype              The type of weight to generate (FLOAT32 or FLOAT64), ignored
 * unless include_weights is true
 * @param [in]     minimum_weight     Minimum weight value to generate
 * @param [in]     maximum_weight     Maximum weight value to generate
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not CUGRAPH_SUCCESS
 */
cugraph_error_code_t cugraph_generate_edge_weights(const cugraph_resource_handle_t* handle,
                                                   cugraph_rng_state_t* rng_state,
                                                   cugraph_coo_t* coo,
                                                   cugraph_data_type_id_t dtype,
                                                   double minimum_weight,
                                                   double maximum_weight,
                                                   cugraph_error_t** error);

/**
 * @brief      Add edge ids to an COO
 *
 * Updates a COO to contain edge ids.  Edges will be numbered from 0 to n-1 where n is the number of
 * edges
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [in/out] coo                Opaque pointer to the coo, weights will be added (overwriting
 * any existing weights)
 * @param [in]     multi_gpu          Flag if the COO is being created on multiple GPUs
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not CUGRAPH_SUCCESS
 */
cugraph_error_code_t cugraph_generate_edge_ids(const cugraph_resource_handle_t* handle,
                                               cugraph_coo_t* coo,
                                               bool_t multi_gpu,
                                               cugraph_error_t** error);

/**
 * @brief      Generate random edge types, add them to an COO
 *
 * Updates a COO to contain edge types.  Edges types will be randomly generated.
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [in/out] rng_state          State of the random number generator, updated with each call
 * @param [in/out] coo                Opaque pointer to the coo, weights will be added (overwriting
 * any existing weights)
 * @param [in]     max_edge_type      Edge types will be randomly generated between min_edge_type
 * and max_edge_type
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not CUGRAPH_SUCCESS
 */
cugraph_error_code_t cugraph_generate_edge_types(const cugraph_resource_handle_t* handle,
                                                 cugraph_rng_state_t* rng_state,
                                                 cugraph_coo_t* coo,
                                                 int32_t min_edge_type,
                                                 int32_t max_edge_type,
                                                 cugraph_error_t** error);

#ifdef __cplusplus
}
#endif
