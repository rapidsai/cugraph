/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

// Author: Prasun Gera pgera@nvidia.com

#pragma once
#include <rmm/device_vector.hpp>

namespace cugraph {
namespace detail {
template <typename IndexType, typename DistType>
class SSSP {
 private:
  IndexType n, nnz;
  const IndexType* row_offsets;
  const IndexType* col_indices;
  const DistType* edge_weights;

  // edgemask, distances, predecessors are set/read by users - using Vectors
  bool useEdgeMask;
  bool computeDistances;
  bool computePredecessors;
  DistType* distances;
  DistType* next_distances;
  rmm::device_vector<DistType> distances_vals;
  rmm::device_vector<DistType> next_distances_vals;
  IndexType* predecessors;
  int* edge_mask;

  // Working data
  IndexType nisolated;
  rmm::device_vector<IndexType> frontier, new_frontier;
  IndexType vertices_bmap_size, edges_bmap_size;
  int *relaxed_edges_bmap, *next_frontier_bmap;
  rmm::device_vector<int> isolated_bmap;
  rmm::device_vector<IndexType> vertex_degree;
  rmm::device_buffer iter_buffer;
  size_t iter_buffer_size;
  rmm::device_vector<IndexType> frontier_vertex_degree;
  rmm::device_vector<IndexType> exclusive_sum_frontier_vertex_degree;
  rmm::device_vector<IndexType> exclusive_sum_frontier_vertex_buckets_offsets;
  IndexType* d_new_frontier_cnt;

  cudaStream_t stream;

  void setup();
  void clean();

 public:
  virtual ~SSSP(void) { clean(); }

  SSSP(IndexType _n,
       IndexType _nnz,
       const IndexType* _row_offsets,
       const IndexType* _col_indices,
       const DistType* _edge_weights,
       cudaStream_t _stream = 0)
    : n(_n),
      nnz(_nnz),
      row_offsets(_row_offsets),
      edge_weights(_edge_weights),
      col_indices(_col_indices),
      stream(_stream)
  {
    setup();
  }

  void configure(DistType* distances, IndexType* predecessors, int* edge_mask);
  void traverse(IndexType source_vertex);
};
}  // namespace detail
}  // namespace cugraph
