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

#include <cugraph.h>
#include "utilities/error_utils.h"
#include <rmm_utils.h>
#include "utilities/graph_utils.cuh"
#include "community/Cluster_Helpers.cuh"
#include "community/Graph_Coarsening.cuh"
#include "nvgraph/include/Louvain_modularity.cuh"
#include "nvgraph/include/Louvain_matching.cuh"

namespace cugraph {
template<typename IdxT, typename ValT>
void leiden(Graph* graph,
            int metric,
            ValT gamma,
            IdxT* leiden_parts,
            IdxT max_level) {
  // Check for error conditions
  CUGRAPH_EXPECTS(graph != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(leiden_parts != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(graph->adjList != nullptr, "Graph must have adjacency list");
  CUGRAPH_EXPECTS(graph->adjList->edge_data != nullptr, "Graph must have weights");

  // Get info about the graph
  IdxT n = graph->adjList->offsets->size - 1;
  IdxT nnz = graph->adjList->indices->size;

  // Initialize levels vector
  std::vector<level_info<IdxT, ValT>> levels;
  levels.push_back(level_info<IdxT, ValT>());
  levels.back().num_verts = n;
  levels.back().nnz = nnz;
  levels.back().num_clusters = n;
  levels.back().csr_off = (IdxT*) graph->adjList->offsets->data;
  levels.back().csr_ind = (IdxT*) graph->adjList->indices->data;
  levels.back().csr_val = (ValT*) graph->adjList->edge_data->data;

  IdxT* cluster_inv_off;
  IdxT* cluster_inv_ind;
  IdxT* clusters;
  ALLOC_TRY(&cluster_inv_off, sizeof(IdxT) * (n + 1), nullptr);
  ALLOC_TRY(&cluster_inv_ind, sizeof(IdxT) * n, nullptr);
  ALLOC_TRY(&clusters, sizeof(IdxT) * n, nullptr);

  levels.back().clusters = clusters;
  levels.back().cluster_inv_off = cluster_inv_off;
  levels.back().cluster_inv_ind = cluster_inv_ind;

  // Assign initial singleton partition
  thrust::sequence(rmm::exec_policy(nullptr)->on(nullptr),
                   levels.back().clusters,
                   levels.back().clusters + n,
                   0);

  // Temporary vectors used for computing metrics
  rmm::device_vector < ValT > k_vec(n, 0);
  rmm::device_vector < ValT > delta_modularity(nnz, 0);
  rmm::device_vector < ValT > e_c(n, 0);
  rmm::device_vector < ValT > k_c(n, 0);
  rmm::device_vector < ValT > m_c(n, 0);
  ValT* e_c_ptr = thrust::raw_pointer_cast(e_c.data());
  ValT* k_c_ptr = thrust::raw_pointer_cast(k_c.data());
  ValT* m_c_ptr = thrust::raw_pointer_cast(m_c.data());
  ValT* k_vec_ptr = thrust::raw_pointer_cast(k_vec.data());
  ValT* delta_modularity_ptr = thrust::raw_pointer_cast(delta_modularity.data());

  // Compute initial metric
  ValT m2 = thrust::reduce(rmm::exec_policy(nullptr)->on(nullptr),
                           levels.back().csr_val,
                           levels.back().csr_val + nnz);
  compute_k_vec(levels.back().num_verts,
                levels.back().csr_off,
                levels.back().csr_val,
                k_vec_ptr);
  ValT initial_modularity = compute_modularity(levels.back().num_verts,
                                               levels.back().nnz,
                                               levels.back().num_clusters,
                                               levels.back().csr_off,
                                               levels.back().csr_ind,
                                               levels.back().csr_val,
                                               k_vec_ptr,
                                               levels.back().clusters,
                                               gamma,
                                               m2,
                                               e_c_ptr,
                                               k_c_ptr,
                                               m_c_ptr);

  std::cout << "Initial modularity value: " << initial_modularity << "\n";

  // Main loop
  while (true) {
    // Compute m2 and k_vec for current level
    ValT m2 = thrust::reduce(rmm::exec_policy(nullptr)->on(nullptr),
                             levels.back().csr_val,
                             levels.back().csr_val + nnz);
    compute_k_vec(levels.back().num_verts,
                  levels.back().csr_off,
                  levels.back().csr_val,
                  k_vec_ptr);

    // Initial swappings
    IdxT num_moved = 1;
    IdxT total_moved = 0;
    while (num_moved > 0) {
      // Compute modularity to initialize e_c, k_c, and m_c vectors
      ValT modularity = compute_modularity(levels.back().num_verts,
                                           levels.back().nnz,
                                           levels.back().num_clusters,
                                           levels.back().csr_off,
                                           levels.back().csr_ind,
                                           levels.back().csr_val,
                                           k_vec_ptr,
                                           levels.back().clusters,
                                           gamma,
                                           m2,
                                           e_c_ptr,
                                           k_c_ptr,
                                           m_c_ptr);
      // Compute the delta modularity vector
      compute_delta_modularity(levels.back().nnz,
                               levels.back().num_verts,
                               levels.back().csr_off,
                               levels.back().csr_ind,
                               levels.back().csr_val,
                               levels.back().clusters,
                               e_c_ptr,
                               k_c_ptr,
                               m_c_ptr,
                               k_vec_ptr,
                               delta_modularity_ptr,
                               gamma,
                               m2);

      // Make swaps
      num_moved = makeSwaps(levels.back().num_verts,
                            levels.back().csr_off,
                            levels.back().csr_ind,
                            delta_modularity_ptr,
                            levels.back().clusters);
      total_moved += num_moved;

      // Renumber and count the resulting aggregates
      renumberAndCountAggregates(levels.back().clusters,
                                 levels.back().num_verts,
                                 levels.back().num_clusters);

    }

    // If we got through the swapping phase without making any swaps we are done
    if (total_moved == 0 && levels.size() == 1)
      break;

    // If we didn't make any swaps, but are not at the first level, project down and keep going
    if (total_moved == 0) {
      while (levels.size() > 1) {
        // Do some magic and project down the last level to the previous level
        level_info<IdxT, ValT>last = levels.back();
        level_info<IdxT, ValT>prev = levels[levels.size() - 2];
        project(last.num_verts,
                prev.cluster_inv_off,
                prev.cluster_inv_ind,
                last.clusters,
                prev.clusters);
        std::cout << "Projected level " << levels.size() << " down\n";
        // Delete the last frame and pop it off
        levels.back().delete_all();
        levels.pop_back();
      }
      std::cout << "Starting to iterate again on projected result:\n";
      continue;
    }

    if ((IdxT) levels.size() >= max_level) {
      generate_cluster_inv(levels.back().num_verts,
                           levels.back().num_clusters,
                           levels.back().clusters,
                           levels.back().cluster_inv_off,
                           levels.back().cluster_inv_ind);
      break;
    }

    // We made swaps so now we refine the partition (Leiden)

    // And now coarsen the graph
    generate_cluster_inv(levels.back().num_verts,
                         levels.back().num_clusters,
                         levels.back().clusters,
                         levels.back().cluster_inv_off,
                         levels.back().cluster_inv_ind);
    level_info<IdxT, ValT>prev = levels.back();
    levels.push_back(level_info<IdxT, ValT>());
    levels.back().num_verts = prev.num_clusters;
    levels.back().num_clusters = prev.num_clusters;

    generate_supervertices_graph(prev.num_verts,
                                 prev.nnz,
                                 prev.num_clusters,
                                 prev.csr_off,
                                 prev.csr_ind,
                                 prev.csr_val,
                                 &(levels.back().csr_off),
                                 &(levels.back().csr_ind),
                                 &(levels.back().csr_val),
                                 prev.clusters,
                                 levels.back().nnz);

    std::cout << "Completed generate super vertices graph: num_verts " << levels.back().num_verts
        << " nnz " << levels.back().nnz << "\n";

    IdxT* new_cluster_inv_off;
    IdxT* new_cluster_inv_ind;
    IdxT* new_clusters;
    ALLOC_TRY(&new_cluster_inv_off, sizeof(IdxT) * (prev.num_clusters + 1), nullptr);
    ALLOC_TRY(&new_cluster_inv_ind, sizeof(IdxT) * prev.num_clusters, nullptr);
    ALLOC_TRY(&new_clusters, sizeof(IdxT) * prev.num_clusters, nullptr);
    levels.back().cluster_inv_off = new_cluster_inv_off;
    levels.back().cluster_inv_ind = new_cluster_inv_ind;
    levels.back().clusters = new_clusters;
    thrust::sequence(rmm::exec_policy(nullptr)->on(nullptr),
                     new_clusters,
                     new_clusters + prev.num_clusters,
                     0);
  }

  // Now that we have finished with the levels, it's time to project the solution back down
  // As we project the solution down we will also be deleting temporary memory allocations from each level
  while (levels.size() > 1) {
    // Do some magic and project down the last level to the previous level
    level_info<IdxT, ValT>last = levels.back();
    level_info<IdxT, ValT>prev = levels[levels.size() - 2];
    project(last.num_verts,
            prev.cluster_inv_off,
            prev.cluster_inv_ind,
            last.clusters,
            prev.clusters);
    std::cout << "Projected level " << levels.size() << " down\n";
    // Delete the last frame and pop it off
    levels.back().delete_all();
    levels.pop_back();
  }
  cudaMemcpy(leiden_parts,
             levels.back().clusters,
             levels.back().num_verts * sizeof(IdxT),
             cudaMemcpyDefault);
  levels.back().delete_added();
}

// Explicit template instantiations
template void leiden<int32_t, float>(Graph* graph,
                                     int metric,
                                     float gamma,
                                     int32_t* leiden_parts,
                                     int32_t max_iter);
template void leiden<int32_t, double>(Graph* graph,
                                      int metric,
                                      double gamma,
                                      int32_t* leiden_parts,
                                      int32_t max_iter);
template void leiden<int64_t, float>(Graph* graph,
                                     int metric,
                                     float gamma,
                                     int64_t* leiden_parts,
                                     int64_t max_iter);
template void leiden<int64_t, double>(Graph* graph,
                                      int metric,
                                      double gamma,
                                      int64_t* leiden_parts,
                                      int64_t max_iter);

} // cugraph namespace
