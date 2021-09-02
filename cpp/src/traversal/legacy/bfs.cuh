/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#pragma once

#include <climits>
#include <rmm/device_vector.hpp>

#define TRAVERSAL_DEFAULT_ALPHA 15

#define TRAVERSAL_DEFAULT_BETA 18

namespace cugraph {
namespace detail {
// FIXME: Differentiate IndexType for vertices and edges
template <typename IndexType>
class BFS {
 private:
  IndexType number_of_vertices, number_of_edges;
  const IndexType* row_offsets = nullptr;
  const IndexType* col_indices = nullptr;

  bool directed;
  bool deterministic;

  // edgemask, distances, predecessors are set/read by users - using Vectors
  bool useEdgeMask;
  bool computeDistances;
  bool computePredecessors;
  rmm::device_vector<IndexType> distances_vals;
  IndexType* distances    = nullptr;
  IndexType* predecessors = nullptr;
  double* sp_counters     = nullptr;
  int* edge_mask          = nullptr;

  rmm::device_vector<IndexType> original_frontier;
  rmm::device_vector<int> visited_bmap;
  rmm::device_vector<int> isolated_bmap;
  rmm::device_vector<int> previous_visited_bmap;
  rmm::device_vector<IndexType> vertex_degree;
  rmm::device_vector<IndexType> buffer_np1_1;
  rmm::device_vector<IndexType> buffer_np1_2;
  rmm::device_vector<IndexType> exclusive_sum_frontier_vertex_buckets_offsets;
  rmm::device_vector<IndexType> d_counters_pad;
  // Working data
  // For complete description of each, go to bfs.cu
  IndexType nisolated;
  IndexType* frontier                             = nullptr;
  IndexType* new_frontier                         = nullptr;
  IndexType* frontier_vertex_degree               = nullptr;
  IndexType* exclusive_sum_frontier_vertex_degree = nullptr;
  IndexType* unvisited_queue                      = nullptr;
  IndexType* left_unvisited_queue                 = nullptr;
  IndexType* d_new_frontier_cnt                   = nullptr;
  IndexType* d_mu                                 = nullptr;
  IndexType* d_unvisited_cnt                      = nullptr;
  IndexType* d_left_unvisited_cnt                 = nullptr;

  IndexType vertices_bmap_size;

  // Parameters for direction optimizing
  IndexType alpha, beta;
  cudaStream_t stream;

  // resets pointers defined by d_counters_pad (see implem)
  void resetDevicePointers();
  void setup();
  void clean();

 public:
  virtual ~BFS(void) { clean(); }

  BFS(IndexType _number_of_vertices,
      IndexType _number_of_edges,
      const IndexType* _row_offsets,
      const IndexType* _col_indices,
      bool _directed,
      IndexType _alpha,
      IndexType _beta,
      cudaStream_t _stream = 0)
    : number_of_vertices(_number_of_vertices),
      number_of_edges(_number_of_edges),
      row_offsets(_row_offsets),
      col_indices(_col_indices),
      directed(_directed),
      alpha(_alpha),
      beta(_beta),
      stream(_stream)
  {
    setup();
  }

  void configure(IndexType* distances,
                 IndexType* predecessors,
                 double* sp_counters,
                 int* edge_mask);

  void traverse(IndexType source_vertex);
};
}  // namespace detail
}  // namespace cugraph
