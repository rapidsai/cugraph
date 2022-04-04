/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cugraph/utilities/error.hpp>
#include <rmm/device_vector.hpp>

#include <cugraph/legacy/graph.hpp>

#include "sssp.cuh"
#include "sssp_kernels.cuh"
#include "traversal_common.cuh"

namespace cugraph {
namespace detail {

template <typename IndexType, typename DistType>
void SSSP<IndexType, DistType>::setup()
{
  // Working data
  // Each vertex can be in the frontier at most once
  frontier.resize(n);
  new_frontier.resize(n);

  // size of bitmaps for vertices
  vertices_bmap_size = (n / (8 * sizeof(int)) + 1);

  // size of bitmaps for edges
  edges_bmap_size = (nnz / (8 * sizeof(int)) + 1);

  // ith bit of isolated_bmap is set <=> degree of ith vertex = 0
  isolated_bmap.resize(vertices_bmap_size);

  // Allocate buffer for data that need to be reset every iteration
  iter_buffer_size = sizeof(int) * (edges_bmap_size + vertices_bmap_size) + sizeof(IndexType);
  iter_buffer.resize(iter_buffer_size, stream);
  // ith bit of relaxed_edges_bmap <=> ith edge was relaxed
  relaxed_edges_bmap = static_cast<int*>(iter_buffer.data());
  // ith bit of next_frontier_bmap <=> vertex is active in the next frontier
  next_frontier_bmap = static_cast<int*>(iter_buffer.data()) + edges_bmap_size;
  // num vertices in the next frontier
  d_new_frontier_cnt = next_frontier_bmap + vertices_bmap_size;

  // vertices_degree[i] = degree of vertex i
  vertex_degree.resize(n);

  // frontier_vertex_degree[i] is the degree of vertex frontier[i]
  frontier_vertex_degree.resize(n);

  // exclusive sum of frontier_vertex_degree
  exclusive_sum_frontier_vertex_degree.resize(n + 1);

  // We use buckets of edges (32 edges per bucket for now, see exact macro in
  // sssp_kernels). frontier_vertex_degree_buckets_offsets[i] is the index k
  // such as frontier[k] is the source of the first edge of the bucket
  // See top down kernels for more details
  size_t bucket_off_size =
    ((nnz / TOP_DOWN_EXPAND_DIMX + 1) * NBUCKETS_PER_BLOCK + 2) * sizeof(IndexType);
  exclusive_sum_frontier_vertex_buckets_offsets.resize(bucket_off_size);

  // Repurpose d_new_frontier_cnt temporarily
  IndexType* d_nisolated = d_new_frontier_cnt;
  cudaMemsetAsync(d_nisolated, 0, sizeof(IndexType), stream);

  // Computing isolated_bmap
  // Only dependent on graph - not source vertex - done once
  traversal::flag_isolated_vertices(
    n, isolated_bmap.data().get(), row_offsets, vertex_degree.data().get(), d_nisolated, stream);

  cudaMemcpyAsync(&nisolated, d_nisolated, sizeof(IndexType), cudaMemcpyDeviceToHost, stream);

  // We need nisolated to be ready to use
  // nisolated is the number of isolated (zero out-degree) vertices
  cudaStreamSynchronize(stream);
}

template <typename IndexType, typename DistType>
void SSSP<IndexType, DistType>::configure(DistType* _distances,
                                          IndexType* _predecessors,
                                          int* _edge_mask)
{
  distances    = _distances;
  predecessors = _predecessors;
  edge_mask    = _edge_mask;

  useEdgeMask         = (edge_mask != NULL);
  computeDistances    = (distances != NULL);
  computePredecessors = (predecessors != NULL);

  // We need distances for SSSP even if the caller doesn't need them
  if (!computeDistances) {
    distances_vals.resize(n);
    distances = distances_vals.data().get();
  }
  // Need next_distances in either case
  next_distances_vals.resize(n);
  next_distances = next_distances_vals.data().get();
}

template <typename IndexType, typename DistType>
void SSSP<IndexType, DistType>::traverse(IndexType source_vertex)
{
  // Init distances to infinities
  traversal::fill_vec(distances, n, traversal::vec_t<DistType>::max, stream);
  traversal::fill_vec(next_distances, n, traversal::vec_t<DistType>::max, stream);

  // If needed, set all predecessors to non-existent (-1)
  if (computePredecessors) { cudaMemsetAsync(predecessors, -1, n * sizeof(IndexType), stream); }

  //
  // Initial frontier
  //

  cudaMemsetAsync(&distances[source_vertex], 0, sizeof(DistType), stream);
  cudaMemsetAsync(&next_distances[source_vertex], 0, sizeof(DistType), stream);

  int current_isolated_bmap_source_vert = 0;

  cudaMemcpyAsync(&current_isolated_bmap_source_vert,
                  isolated_bmap.data().get() + (source_vertex / INT_SIZE),
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
  cudaMemcpyAsync(
    frontier.data().get(), &source_vertex, sizeof(IndexType), cudaMemcpyHostToDevice, stream);

  // Number of vertices in the frontier and number of out-edges from the
  // frontier
  IndexType mf, nf;
  nf        = 1;
  int iters = 0;

  while (nf > 0) {
    // Typical pre-top down workflow. set_frontier_degree + exclusive-scan
    traversal::set_frontier_degree(frontier_vertex_degree.data().get(),
                                   frontier.data().get(),
                                   vertex_degree.data().get(),
                                   nf,
                                   stream);

    traversal::exclusive_sum(frontier_vertex_degree.data().get(),
                             exclusive_sum_frontier_vertex_degree.data().get(),
                             nf + 1,
                             stream);

    cudaMemcpyAsync(&mf,
                    exclusive_sum_frontier_vertex_degree.data().get() + nf,
                    sizeof(IndexType),
                    cudaMemcpyDeviceToHost,
                    stream);

    // We need mf to know the next kernel's launch dims
    cudaStreamSynchronize(stream);

    traversal::compute_bucket_offsets(exclusive_sum_frontier_vertex_degree.data().get(),
                                      exclusive_sum_frontier_vertex_buckets_offsets.data().get(),
                                      nf,
                                      mf,
                                      stream);

    // Reset the transient structures to 0
    cudaMemsetAsync(iter_buffer.data(), 0, iter_buffer_size, stream);

    sssp_kernels::frontier_expand(row_offsets,
                                  col_indices,
                                  edge_weights,
                                  frontier.data().get(),
                                  nf,
                                  mf,
                                  new_frontier.data().get(),
                                  d_new_frontier_cnt,
                                  exclusive_sum_frontier_vertex_degree.data().get(),
                                  exclusive_sum_frontier_vertex_buckets_offsets.data().get(),
                                  distances,
                                  next_distances,
                                  predecessors,
                                  edge_mask,
                                  next_frontier_bmap,
                                  relaxed_edges_bmap,
                                  isolated_bmap.data().get(),
                                  stream);

    cudaMemcpyAsync(&nf, d_new_frontier_cnt, sizeof(IndexType), cudaMemcpyDeviceToHost, stream);

    // Copy next_distances to distances
    cudaMemcpyAsync(
      distances, next_distances, n * sizeof(DistType), cudaMemcpyDeviceToDevice, stream);

    // We need nf for the loop
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    // Swap frontiers
    // IndexType *tmp = frontier;
    // frontier       = new_frontier;
    // new_frontier   = tmp;
    new_frontier.swap(frontier);
    iters++;

    if (iters > n) {
      // Bail out. Got a graph with a negative cycle
      CUGRAPH_FAIL("ERROR: Max iterations exceeded. Check the graph for negative weight cycles");
    }
  }
}

template <typename IndexType, typename DistType>
void SSSP<IndexType, DistType>::clean()
{
}

}  // namespace detail

/**
 * ---------------------------------------------------------------------------*
 * @brief Native sssp with predecessors
 *
 * @file sssp.cu
 * --------------------------------------------------------------------------*/
template <typename VT, typename ET, typename WT>
void sssp(legacy::GraphCSRView<VT, ET, WT> const& graph,
          WT* distances,
          VT* predecessors,
          const VT source_vertex)
{
  CUGRAPH_EXPECTS(distances || predecessors, "Invalid input argument, both outputs are nullptr");

  if (typeid(VT) != typeid(int)) CUGRAPH_FAIL("Unsupported vertex id data type, please use int");
  if (typeid(ET) != typeid(int)) CUGRAPH_FAIL("Unsupported edge id data type, please use int");
  if (typeid(WT) != typeid(float) && typeid(WT) != typeid(double))
    CUGRAPH_FAIL("Unsupported weight data type, please use float or double");

  int num_vertices = graph.number_of_vertices;
  int num_edges    = graph.number_of_edges;

  const ET* offsets_ptr      = graph.offsets;
  const VT* indices_ptr      = graph.indices;
  const WT* edge_weights_ptr = nullptr;

  // Both if / else branch operate own calls due to
  // thrust::device_vector lifetime
  if (!graph.edge_data) {
    // Generate unit weights

    // FIXME: This should fallback to BFS, but for now it'll go through the
    // SSSP path since BFS needs the directed flag, which should not be
    // necessary for the SSSP API. We can pass directed to the BFS call, but
    // BFS also does only integer distances right now whereas we need float or
    // double

    rmm::device_vector<WT> d_edge_weights(num_edges, static_cast<WT>(1));
    edge_weights_ptr = thrust::raw_pointer_cast(&d_edge_weights.front());
    cugraph::detail::SSSP<VT, WT> sssp(
      num_vertices, num_edges, offsets_ptr, indices_ptr, edge_weights_ptr);
    sssp.configure(distances, predecessors, nullptr);
    sssp.traverse(source_vertex);
  } else {
    // SSSP is not defined for graphs with negative weight cycles
    // Warn user about any negative edges
    if (graph.prop.has_negative_edges == legacy::PropType::PROP_TRUE)
      std::cerr << "WARN: The graph has negative weight edges. SSSP will not "
                   "converge if the graph has negative weight cycles\n";
    edge_weights_ptr = graph.edge_data;
    cugraph::detail::SSSP<VT, WT> sssp(
      num_vertices, num_edges, offsets_ptr, indices_ptr, edge_weights_ptr);
    sssp.configure(distances, predecessors, nullptr);
    sssp.traverse(source_vertex);
  }
}

// explicit instantiation
template void sssp<int, int, float>(legacy::GraphCSRView<int, int, float> const& graph,
                                    float* distances,
                                    int* predecessors,
                                    const int source_vertex);
template void sssp<int, int, double>(legacy::GraphCSRView<int, int, double> const& graph,
                                     double* distances,
                                     int* predecessors,
                                     const int source_vertex);

}  // namespace cugraph
