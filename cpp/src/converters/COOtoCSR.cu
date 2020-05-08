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

#include <functions.hpp>
#include "COOtoCSR.cuh"

namespace cugraph {

template <typename vertex_t, typename edge_t>
vertex_t coo2csr(
  edge_t num_edges, vertex_t const *src, vertex_t const *dst, edge_t **offsets, vertex_t **indices)
{
  CSR_Result<vertex_t> result;
  ConvertCOOtoCSR(src, dst, num_edges, result);

  *offsets = result.rowOffsets;
  *indices = result.colIndices;
  return result.size;
}

template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t coo2csr_weighted(edge_t num_edges,
                          vertex_t const *src,
                          vertex_t const *dst,
                          weight_t const *weights,
                          edge_t **offsets,
                          vertex_t **indices,
                          weight_t **csr_weights)
{
  CSR_Result_Weighted<vertex_t, weight_t> result;
  ConvertCOOtoCSR_weighted(src, dst, weights, num_edges, result);

  *offsets     = result.rowOffsets;
  *indices     = result.colIndices;
  *csr_weights = result.edgeWeights;

  return result.size;
}

template int32_t coo2csr<int32_t, int32_t>(
  int32_t, int32_t const *, int32_t const *, int32_t **, int32_t **);
template int32_t coo2csr_weighted<int32_t, int32_t, float>(
  int32_t, int32_t const *, int32_t const *, float const *, int32_t **, int32_t **, float **);
template int32_t coo2csr_weighted<int32_t, int32_t, double>(
  int32_t, int32_t const *, int32_t const *, double const *, int32_t **, int32_t **, double **);

template std::unique_ptr<experimental::GraphCSR<int32_t, int32_t, float>>
coo_to_csr<int32_t, int32_t, float>(
  experimental::GraphCOOView<int32_t, int32_t, float> const &graph,
  rmm::mr::device_memory_resource *);
template std::unique_ptr<experimental::GraphCSR<int32_t, int32_t, double>>
coo_to_csr<int32_t, int32_t, double>(
  experimental::GraphCOOView<int32_t, int32_t, double> const &graph,
  rmm::mr::device_memory_resource *);

}  // namespace cugraph
