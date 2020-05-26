/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <rmm/device_buffer.hpp>

#include <graph.hpp>

namespace cugraph {

/**
 * @brief    Convert COO to CSR, unweighted
 *
 * Takes a list of edges in COOrdinate format and generates a CSR format.
 * Note, if you want CSC format simply pass the src and dst arrays
 * in the opposite order.
 *
 * @throws                    cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t           type of vertex index
 * @tparam edge_t             type of edge index
 *
 * @param[in]  num_edges      Number of edges
 * @param[in]  src            Device array containing original source vertices
 * @param[in]  dst            Device array containing original dest vertices
 * @param[out] offsets        Device array containing the CSR offsets
 * @param[out] indices        Device array containing the CSR indices
 *
 * @return                    Number of unique vertices in the src and dst arrays
 *
 */
template <typename vertex_t, typename edge_t>
vertex_t coo2csr(
  edge_t num_edges, vertex_t const *src, vertex_t const *dst, edge_t **offsets, vertex_t **indices);

/**
 * @brief    Convert COO to CSR, weighted
 *
 * Takes a list of edges in COOrdinate format and generates a CSR format.
 * Note, if you want CSC format simply pass the src and dst arrays
 * in the opposite order.
 *
 * @throws                    cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t           type of vertex index
 * @tparam edge_t             type of edge index
 * @tparam weight_t           type of the edge weight
 *
 * @param[in]  num_edges      Number of edges
 * @param[in]  src            Device array containing original source vertices
 * @param[in]  dst            Device array containing original dest vertices
 * @param[in]  weights        Device array containing original edge weights
 * @param[out] offsets        Device array containing the CSR offsets
 * @param[out] indices        Device array containing the CSR indices
 * @param[out] csr_weights    Device array containing the CSR edge weights
 *
 * @return                    Number of unique vertices in the src and dst arrays
 *
 */
template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t coo2csr_weighted(edge_t num_edges,
                          vertex_t const *src,
                          vertex_t const *dst,
                          weight_t const *weights,
                          edge_t **offsets,
                          vertex_t **indices,
                          weight_t **csr_weights);

/**
 * @brief    Convert COO to CSR
 *
 * Takes a list of edges in COOrdinate format and generates a CSR format.
 *
 * @throws                    cugraph::logic_error when an error occurs.
 *
 * @tparam VT                 type of vertex index
 * @tparam ET                 type of edge index
 * @tparam WT                 type of the edge weight
 *
 * @param[in]  graph          cuGRAPH graph in coordinate format
 * @param[in]  mr             Memory resource used to allocate the returned graph
 *
 * @return                    Unique pointer to generate Compressed Sparse Row graph
 *
 */
template <typename VT, typename ET, typename WT>
std::unique_ptr<experimental::GraphCSR<VT, ET, WT>> coo_to_csr(
  experimental::GraphCOOView<VT, ET, WT> const &graph,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

/**
 * @brief    Renumber source and destination indices
 *
 * Renumber source and destination indexes to be a dense numbering,
 * using contiguous values between 0 and number of vertices minus 1.
 *
 * @throws                    cugraph::logic_error when an error occurs.
 *
 * @tparam VT_IN              type of vertex index input
 * @tparam VT_OUT             type of vertex index output
 * @tparam ET                 type of edge index
 *
 * @param[in]  number_of_edges number of edges in the graph
 * @param[in]  src            Pointer to device memory containing source vertex ids
 * @param[in]  dst            Pointer to device memory containing destination vertex ids
 * @param[out] src_renumbered Pointer to device memory containing the output source vertices.
 * @param[out] dst_renumbered Pointer to device memory containing the output destination vertices.
 * @param[out] map_size       Pointer to local memory containing the number of elements in the
 * renumbering map
 * @param[in]  mr             Memory resource used to allocate the returned graph
 *
 * @return                    Unique pointer to renumbering map
 *
 */
template <typename VT_IN, typename VT_OUT, typename ET>
std::unique_ptr<rmm::device_buffer> renumber_vertices(
  ET number_of_edges,
  VT_IN const *src,
  VT_IN const *dst,
  VT_OUT *src_renumbered,
  VT_OUT *dst_renumbered,
  ET *map_size,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

}  // namespace cugraph
