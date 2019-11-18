/*
 * Copyright (c) 2019 NVIDIA CORPORATION.
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

namespace cugraph { 
namespace detail {
template <typename IndexType, typename DistType>
class SSSP {
 private:
  IndexType n, nnz;
  IndexType* row_offsets;
  IndexType* col_indices;
  DistType* edge_weights;

  // edgemask, distances, predecessors are set/read by users - using Vectors
  bool useEdgeMask;
  bool computeDistances;
  bool computePredecessors;
  DistType* distances;
  DistType* next_distances;
  IndexType* predecessors;
  int* edge_mask;

  // Working data
  IndexType nisolated;
  IndexType *frontier, *new_frontier;
  IndexType vertices_bmap_size, edges_bmap_size;
  int *isolated_bmap, *relaxed_edges_bmap, *next_frontier_bmap;
  IndexType* vertex_degree;
  void* iter_buffer;
  size_t iter_buffer_size;
  IndexType* frontier_vertex_degree;
  IndexType* exclusive_sum_frontier_vertex_degree;
  IndexType* exclusive_sum_frontier_vertex_buckets_offsets;
  IndexType* d_new_frontier_cnt;
  void* d_cub_exclusive_sum_storage;
  size_t cub_exclusive_sum_storage_bytes;

  cudaStream_t stream;

  void setup();
  void clean();

 public:
  virtual ~SSSP(void) { clean(); }

  SSSP(IndexType _n,
       IndexType _nnz,
       IndexType* _row_offsets,
       IndexType* _col_indices,
       DistType* _edge_weights,
       cudaStream_t _stream = 0)
      : n(_n),
        nnz(_nnz),
        row_offsets(_row_offsets),
        edge_weights(_edge_weights),
        col_indices(_col_indices),
        stream(_stream) {
    setup();
  }

  void configure(DistType* distances, IndexType* predecessors, int* edge_mask);
  void traverse(IndexType source_vertex);
};
} } //namespace
