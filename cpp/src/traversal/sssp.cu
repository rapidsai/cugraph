/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cugraph.h>
#include <rmm_utils.h>
#include <algorithm>

#include "traversal_common.cuh"
#include "sssp.cuh"
#include "sssp_kernels.cuh"
#include "utilities/error_utils.h"

namespace cugraph { 
namespace detail {

template <typename IndexType, typename DistType>
void SSSP<IndexType, DistType>::setup() {
  // Working data
  // Each vertex can be in the frontier at most once
  ALLOC_TRY(&frontier, n * sizeof(IndexType), nullptr);
  ALLOC_TRY(&new_frontier, n * sizeof(IndexType), nullptr);

  // size of bitmaps for vertices
  vertices_bmap_size = (n / (8 * sizeof(int)) + 1);

  // size of bitmaps for edges
  edges_bmap_size = (nnz / (8 * sizeof(int)) + 1);

  // ith bit of isolated_bmap is set <=> degree of ith vertex = 0
  ALLOC_TRY(&isolated_bmap, sizeof(int) * vertices_bmap_size, nullptr);

  // Allocate buffer for data that need to be reset every iteration
  iter_buffer_size =
      sizeof(int) * (edges_bmap_size + vertices_bmap_size) + sizeof(IndexType);
  ALLOC_TRY(&iter_buffer, iter_buffer_size, nullptr);
  // ith bit of relaxed_edges_bmap <=> ith edge was relaxed
  relaxed_edges_bmap = (int*)iter_buffer;
  // ith bit of next_frontier_bmap <=> vertex is active in the next frontier
  next_frontier_bmap = (int*)iter_buffer + edges_bmap_size;
  // num vertices in the next frontier
  d_new_frontier_cnt = next_frontier_bmap + vertices_bmap_size;

  // vertices_degree[i] = degree of vertex i
  ALLOC_TRY(&vertex_degree, sizeof(IndexType) * n, nullptr);

  // Cub working data
  traversal::cub_exclusive_sum_alloc(
      n + 1, d_cub_exclusive_sum_storage, cub_exclusive_sum_storage_bytes);

  // frontier_vertex_degree[i] is the degree of vertex frontier[i]
  ALLOC_TRY(&frontier_vertex_degree, n * sizeof(IndexType), nullptr);

  // exclusive sum of frontier_vertex_degree
  ALLOC_TRY(&exclusive_sum_frontier_vertex_degree,
            (n + 1) * sizeof(IndexType),
            nullptr);

  // We use buckets of edges (32 edges per bucket for now, see exact macro in
  // sssp_kernels). frontier_vertex_degree_buckets_offsets[i] is the index k
  // such as frontier[k] is the source of the first edge of the bucket
  // See top down kernels for more details
  size_t bucket_off_size =
      ((nnz / TOP_DOWN_EXPAND_DIMX + 1) * NBUCKETS_PER_BLOCK + 2) *
      sizeof(IndexType);
  ALLOC_TRY(&exclusive_sum_frontier_vertex_buckets_offsets,
            bucket_off_size,
            nullptr);

  // Repurpose d_new_frontier_cnt temporarily
  IndexType* d_nisolated = d_new_frontier_cnt;
  cudaMemsetAsync(d_nisolated, 0, sizeof(IndexType), stream);

  // Computing isolated_bmap
  // Only dependent on graph - not source vertex - done once
  traversal::flag_isolated_vertices(
      n, isolated_bmap, row_offsets, vertex_degree, d_nisolated, stream);

  cudaMemcpyAsync(&nisolated,
                  d_nisolated,
                  sizeof(IndexType),
                  cudaMemcpyDeviceToHost,
                  stream);

  // We need nisolated to be ready to use
  // nisolated is the number of isolated (zero out-degree) vertices
  cudaStreamSynchronize(stream);
}

template <typename IndexType, typename DistType>
void SSSP<IndexType, DistType>::configure(DistType* _distances,
                                          IndexType* _predecessors,
                                          int* _edge_mask) {
  distances = _distances;
  predecessors = _predecessors;
  edge_mask = _edge_mask;

  useEdgeMask = (edge_mask != NULL);
  computeDistances = (distances != NULL);
  computePredecessors = (predecessors != NULL);

  // We need distances for SSSP even if the caller doesn't need them
  if (!computeDistances)
    ALLOC_TRY(&distances, n * sizeof(DistType), nullptr);
  // Need next_distances in either case
  ALLOC_TRY(&next_distances, n * sizeof(DistType), nullptr);
}

template <typename IndexType, typename DistType>
void SSSP<IndexType, DistType>::traverse(IndexType source_vertex) {
  // Init distances to infinities
  traversal::fill_vec(distances, n, traversal::vec_t<DistType>::max, stream);
  traversal::fill_vec(
      next_distances, n, traversal::vec_t<DistType>::max, stream);

  // If needed, set all predecessors to non-existent (-1)
  if (computePredecessors) {
    cudaMemsetAsync(predecessors, -1, n * sizeof(IndexType), stream);
  }

  //
  // Initial frontier
  //

  cudaMemsetAsync(&distances[source_vertex], 0, sizeof(DistType), stream);
  cudaMemsetAsync(&next_distances[source_vertex], 0, sizeof(DistType), stream);

  int current_isolated_bmap_source_vert = 0;

  cudaMemcpyAsync(&current_isolated_bmap_source_vert,
                  &isolated_bmap[source_vertex / INT_SIZE],
                  sizeof(int),
                  cudaMemcpyDeviceToHost);

  // We need current_isolated_bmap_source_vert
  cudaStreamSynchronize(stream);

  int m = (1 << (source_vertex % INT_SIZE));

  // If source is isolated (zero outdegree), we are done
  if ((m & current_isolated_bmap_source_vert)) {
    // Init distances and predecessors are done; stream is synchronized
    
  }

  // Adding source_vertex to init frontier
  cudaMemcpyAsync(&frontier[0],
                  &source_vertex,
                  sizeof(IndexType),
                  cudaMemcpyHostToDevice,
                  stream);

  // Number of vertices in the frontier and number of out-edges from the
  // frontier
  IndexType mf, nf;
  nf = 1;
  int iters = 0;

  while (nf > 0) {
    // Typical pre-top down workflow. set_frontier_degree + exclusive-scan
    traversal::set_frontier_degree(
        frontier_vertex_degree, frontier, vertex_degree, nf, stream);

    traversal::exclusive_sum(d_cub_exclusive_sum_storage,
                             cub_exclusive_sum_storage_bytes,
                             frontier_vertex_degree,
                             exclusive_sum_frontier_vertex_degree,
                             nf + 1,
                             stream);

    cudaMemcpyAsync(&mf,
                    &exclusive_sum_frontier_vertex_degree[nf],
                    sizeof(IndexType),
                    cudaMemcpyDeviceToHost,
                    stream);

    // We need mf to know the next kernel's launch dims
    cudaStreamSynchronize(stream);

    traversal::compute_bucket_offsets(
        exclusive_sum_frontier_vertex_degree,
        exclusive_sum_frontier_vertex_buckets_offsets,
        nf,
        mf,
        stream);

    // Reset the transient structures to 0
    cudaMemsetAsync(iter_buffer, 0, iter_buffer_size, stream);

    sssp_kernels::frontier_expand(
        row_offsets,
        col_indices,
        edge_weights,
        frontier,
        nf,
        mf,
        new_frontier,
        d_new_frontier_cnt,
        exclusive_sum_frontier_vertex_degree,
        exclusive_sum_frontier_vertex_buckets_offsets,
        distances,
        next_distances,
        predecessors,
        edge_mask,
        next_frontier_bmap,
        relaxed_edges_bmap,
        isolated_bmap,
        stream);

    cudaMemcpyAsync(&nf,
                    d_new_frontier_cnt,
                    sizeof(IndexType),
                    cudaMemcpyDeviceToHost,
                    stream);

    // Copy next_distances to distances
    cudaMemcpyAsync(distances,
                    next_distances,
                    n * sizeof(DistType),
                    cudaMemcpyDeviceToDevice,
                    stream);

    CUDA_CHECK_LAST();

    // We need nf for the loop
    cudaStreamSynchronize(stream);

    // Swap frontiers
    IndexType* tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
    iters++;

    if (iters > n) {
      // Bail out. Got a graph with a negative cycle
      CUGRAPH_FAIL("ERROR: Max iterations exceeded. Check the graph for negative weight cycles");
    }
  }
  
}

template <typename IndexType, typename DistType>
void SSSP<IndexType, DistType>::clean() {
  // the vectors have a destructor that takes care of cleaning
  ALLOC_FREE_TRY(frontier, nullptr);
  ALLOC_FREE_TRY(new_frontier, nullptr);
  ALLOC_FREE_TRY(isolated_bmap, nullptr);
  ALLOC_FREE_TRY(vertex_degree, nullptr);
  ALLOC_FREE_TRY(d_cub_exclusive_sum_storage, nullptr);
  ALLOC_FREE_TRY(frontier_vertex_degree, nullptr);
  ALLOC_FREE_TRY(exclusive_sum_frontier_vertex_degree, nullptr);
  ALLOC_FREE_TRY(exclusive_sum_frontier_vertex_buckets_offsets, nullptr);
  ALLOC_FREE_TRY(iter_buffer, nullptr);

  // Distances were working data
  if (!computeDistances)
    ALLOC_FREE_TRY(distances, nullptr);

  // next_distances were working data
  ALLOC_FREE_TRY(next_distances, nullptr);
}

} //namespace

/**
 * ---------------------------------------------------------------------------*
 * @brief Native sssp with predecessors
 *
 * @file sssp.cu
 * --------------------------------------------------------------------------*/
template <typename VT, typename WT>
void sssp(Graph* graph,                                            
          WT *distances,                                                  
          VT *predecessors,                                               
          const VT source_vertex) {

  CUGRAPH_EXPECTS(graph->adjList != nullptr, "Invalid API parameter");

  CUGRAPH_EXPECTS(distances || predecessors, "Invalid API parameter, both outputs are nullptr");

  if (typeid(WT) != typeid(float) && typeid(WT) != typeid(double))
    CUGRAPH_FAIL("Unsupported weight data type, please use float or double");
  if (typeid(VT) != typeid(int) )
    CUGRAPH_FAIL("Unsupported personalization_subset data type, please use int");

  // TODO fix me after gdf_column is removed from Graph
  CUGRAPH_EXPECTS(graph->adjList->offsets->dtype == GDF_INT32,
              "Unsupported data type");
  CUGRAPH_EXPECTS(graph->adjList->indices->dtype == GDF_INT32,
              "Unsupported data type");
  // TODO fix me after gdf_column is removed from Graph
  // if (predecessors) CUGRAPH_EXPECTS(typeid(predecessors) == typeid(graph->adjList->indices), "predecessors and ID type mismatch");

  if (!graph->adjList->edge_data) {
    // Generate unit weights

    // TODO: This should fallback to BFS, but for now it'll go through the
    // SSSP path since BFS needs the directed flag, which should not be
    // necessary for the SSSP API. We can pass directed to the BFS call, but
    // BFS also does only integer distances right now whereas we need float or
    // double

    void* d_edge_data;
    graph->adjList->edge_data = new gdf_column;
    cudaStream_t stream{nullptr};

    // If distances array is given and is double, generate the weights in
    // double
    if (distances && typeid(WT) == typeid(double)) {
      std::vector<double> h_edge_data(graph->adjList->indices->size, 1.0);
      size_t edge_data_size = sizeof(double) * h_edge_data.size();
      ALLOC_TRY((void**)&d_edge_data, edge_data_size, stream);
      CUDA_TRY(cudaMemcpy(d_edge_data,
                          &h_edge_data[0],
                          edge_data_size,
                          cudaMemcpyHostToDevice));
      gdf_column_view(graph->adjList->edge_data,
                      d_edge_data,
                      nullptr,
                      graph->adjList->indices->size,
                      GDF_FLOAT64);

    } else {
      // Else generate float
      std::vector<float> h_edge_data(graph->adjList->indices->size, 1.0);
      size_t edge_data_size = sizeof(float) * h_edge_data.size();
      ALLOC_TRY((void**)&d_edge_data, edge_data_size, stream);
      CUDA_TRY(cudaMemcpy(d_edge_data,
                          &h_edge_data[0],
                          edge_data_size,
                          cudaMemcpyHostToDevice));
      gdf_column_view(graph->adjList->edge_data,
                      d_edge_data,
                      nullptr,
                      graph->adjList->indices->size,
                      GDF_FLOAT32);
    }
  } else {
    // Got weighted graph
    CUGRAPH_EXPECTS(
        graph->adjList->edge_data->size == graph->adjList->indices->size,
        "Graph sizes mismatch");
    // TODO fix me after gdf_column is removed from Graph
    CUGRAPH_EXPECTS(graph->adjList->edge_data->dtype == GDF_FLOAT32 ||
                    graph->adjList->edge_data->dtype == GDF_FLOAT64,
                "Invalid API parameter");
    // TODO fix me after gdf_column is removed from Graph
    // if (distances) CUGRAPH_EXPECTS(typeid(distances) == typeid(graph->adjList->edge_data), "distances and weights type mismatch");

    // SSSP is not defined for graphs with negative weight cycles
    // Warn user about any negative edges
    if (graph->prop && graph->prop->has_negative_edges == GDF_PROP_TRUE)
      std::cerr << "WARN: The graph has negative weight edges. SSSP will not "
                   "converge if the graph has negative weight cycles\n";
  }

  int n = graph->adjList->offsets->size - 1;
  int e = graph->adjList->indices->size;
  int* offsets_ptr = (int*)graph->adjList->offsets->data;
  int* indices_ptr = (int*)graph->adjList->indices->data;

  void* edge_weights_ptr = static_cast<void*>(graph->adjList->edge_data->data);

  if (typeid(WT) == typeid(float)) {
    cugraph::detail::SSSP<VT, WT> sssp(
        n, e, offsets_ptr, indices_ptr, static_cast<WT*>(edge_weights_ptr));

    sssp.configure(distances, predecessors, nullptr);
    sssp.traverse(source_vertex);
  } else if (typeid(WT) == typeid(double)) {
    cugraph::detail::SSSP<VT, WT> sssp(n,
                                    e,
                                    offsets_ptr,
                                    indices_ptr,
                                    static_cast<WT*>(edge_weights_ptr));

    sssp.configure(distances, predecessors, nullptr);
    sssp.traverse(source_vertex);
  } else {
    CUGRAPH_EXPECTS(graph->adjList->edge_data->dtype == GDF_FLOAT32 ||
                    graph->adjList->edge_data->dtype == GDF_FLOAT64,
                "Invalid API parameter");
  }
}

// explicit instantiation
template void sssp<int, float>(Graph* graph, float *distances, int *predecessors, const int source_vertex);
template void sssp<int, double>(Graph* graph, double *distances, int *predecessors, const int source_vertex);

} //namespace