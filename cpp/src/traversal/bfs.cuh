/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
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
#include "rmm_utils.h"

#define TRAVERSAL_DEFAULT_ALPHA 15

#define TRAVERSAL_DEFAULT_BETA 18

namespace cugraph {
namespace detail {
// FIXME: Differentiate IndexType for vertices and edges
template <typename IndexType>
class BFS {
 private:
  IndexType number_of_vertices;
  IndexType number_of_edges;
  const IndexType *row_offsets = nullptr;
  const IndexType *col_indices = nullptr;

  bool directed;
  bool deterministic;

  // edgemask, distances, predecessors are set/read by users - using Vectors
  bool useEdgeMask;
  bool computeDistances;
  bool computePredecessors;
  IndexType *distances    = nullptr;
  IndexType *predecessors = nullptr;
  double *sp_counters     = nullptr;
  int *edge_mask          = nullptr;

  // Working data
  // For complete description of each, go to bfs.cu

  IndexType nisolated;
  // Device vectors
  rmm::device_vector<IndexType> frontier_vec;
  rmm::device_vector<int> visited_bmap_vec;
  rmm::device_vector<int> previous_visited_bmap_vec;
  rmm::device_vector<int> isolated_bmap_vec;
  rmm::device_vector<IndexType> vertex_degree_vec;
  rmm::device_vector<IndexType> buffer_np1_1_vec;
  rmm::device_vector<IndexType> buffer_np1_2_vec;
  rmm::device_vector<IndexType> d_counters_pad_vec;

  rmm::device_vector<IndexType> distances_vec;
  rmm::device_vector<IndexType> exclusive_sum_frontier_vertex_buckets_offsets_vec;

  // Pointers
  IndexType *frontier                                      = nullptr;
  IndexType *new_frontier                                  = nullptr;
  IndexType *original_frontier                             = nullptr;
  int *visited_bmap                                        = nullptr;
  int *isolated_bmap                                       = nullptr;
  int *previous_visited_bmap                               = nullptr;
  IndexType *vertex_degree                                 = nullptr;
  IndexType *buffer_np1_1                                  = nullptr;
  IndexType *buffer_np1_2                                  = nullptr;
  IndexType *frontier_vertex_degree                        = nullptr;
  IndexType *exclusive_sum_frontier_vertex_degree          = nullptr;
  IndexType *unvisited_queue                               = nullptr;
  IndexType *left_unvisited_queue                          = nullptr;
  IndexType *exclusive_sum_frontier_vertex_buckets_offsets = nullptr;
  IndexType *d_counters_pad                                = nullptr;
  IndexType *d_new_frontier_cnt                            = nullptr;
  IndexType *d_mu                                          = nullptr;
  IndexType *d_unvisited_cnt                               = nullptr;
  IndexType *d_left_unvisited_cnt                          = nullptr;
  void *d_cub_exclusive_sum_storage                        = nullptr;

  IndexType vertices_bmap_size;
  size_t cub_exclusive_sum_storage_bytes;
  size_t exclusive_sum_frontier_vertex_buckets_offsets_size;
  size_t d_counters_pad_size;

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
      const IndexType *_row_offsets,
      const IndexType *_col_indices,
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

  void configure(IndexType *distances,
                 IndexType *predecessors,
                 double *sp_counters,
                 int *edge_mask);

  void traverse(IndexType source_vertex);
};
}  // namespace detail
}  // namespace cugraph
