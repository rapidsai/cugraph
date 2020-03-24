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
vertex_t coo2csr(edge_t num_edges,
                 vertex_t const *src,
                 vertex_t const *dst,
                 edge_t **offsets,
                 vertex_t **indices);

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

} //namespace cugraph
