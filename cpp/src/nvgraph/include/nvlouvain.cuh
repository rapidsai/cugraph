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
#pragma once
#include <string>
#include <cstring>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>

#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cusparse.h>

#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include <rmm_utils.h>

#include "graph_utils.cuh"
#include "high_res_clock.h"
#include "Louvain_matching.cuh"
#include "Louvain_modularity.cuh"
#include "community/Cluster_Helpers.cuh"
#include "community/Graph_Coarsening.cuh"
#include "util.cuh"

namespace nvlouvain {

#define VERBOSE true
#define ENABLE_LOG

#define LOG() (log<<COLOR_GRN<<"[ "<< time_now() <<" ] "<<COLOR_WHT)

/*
 The main program of louvain
 */
template<typename IdxType = int, typename ValType>
NVLOUVAIN_STATUS louvain(IdxType* csr_ptr,
                         IdxType* csr_ind,
                         ValType* csr_val,
                         const size_t num_vertex,
                         const size_t num_edges,
                         bool& weighted,
                         bool has_init_cluster,
                         IdxType* init_cluster, // size = n_vertex
                         ValType& final_modularity,
                         IdxType* cluster_vec, // size = n_vertex
                         IdxType& num_level,
                         IdxType max_iter = 100,
                         std::ostream& log = std::cout) {
#ifndef ENABLE_LOG
  log.setstate(std::ios_base::failbit);
#endif
  num_level = 0;
//  cusparseHandle_t cusp_handle;
//  cusparseCreate(&cusp_handle);

  int n_edges = num_edges;
  int n_vertex = num_vertex;

  std::vector<level_info<IdxType, ValType>> levels;
  levels.push_back(level_info<IdxType, ValType>());
  levels.back().num_verts = num_vertex;
  levels.back().nnz = num_edges;
  levels.back().num_clusters = num_vertex;
  levels.back().csr_off = csr_ptr;
  levels.back().csr_ind = csr_ind;
  levels.back().csr_val = csr_val;

  IdxType* cluster_inv_off;
  IdxType* cluster_inv_ind;
  IdxType* clusters;
  ALLOC_TRY(&cluster_inv_off, sizeof(IdxType) * (num_vertex + 1), nullptr);
  ALLOC_TRY(&cluster_inv_ind, sizeof(IdxType) * num_vertex, nullptr);
  ALLOC_TRY(&clusters, sizeof(IdxType) * num_vertex, nullptr);

  levels.back().clusters = clusters;
  levels.back().cluster_inv_off = cluster_inv_off;
  levels.back().cluster_inv_ind = cluster_inv_ind;

  HighResClock hr_clock;
  double timed, diff_time;
  ValType m2 = thrust::reduce(rmm::exec_policy(nullptr)->on(nullptr),
                              levels.back().csr_val,
                              levels.back().csr_val + n_edges);

  rmm::device_vector<ValType> k_vec(n_vertex, 0);
  rmm::device_vector<ValType> delta_Q_arr(n_edges, 0);

  rmm::device_vector<ValType> e_c(n_vertex, 0);
  rmm::device_vector<ValType> k_c(n_vertex, 0);
  rmm::device_vector<ValType> m_c(n_vertex, 0);
  ValType* e_c_ptr = thrust::raw_pointer_cast(e_c.data());
  ValType* k_c_ptr = thrust::raw_pointer_cast(k_c.data());
  ValType* m_c_ptr = thrust::raw_pointer_cast(m_c.data());
  ValType* k_vec_ptr = thrust::raw_pointer_cast(k_vec.data());
  ValType* delta_Q_arr_ptr = thrust::raw_pointer_cast(delta_Q_arr.data());

  if (!has_init_cluster) {
    // if there is no initialized cluster
    // the cluster as assigned as a sequence (a cluster for each vertex)
    // inv_clusters will also be 2 sequence
    thrust::sequence(rmm::exec_policy(nullptr)->on(nullptr),
                     levels.back().clusters,
                     levels.back().clusters + levels.back().num_verts);
    thrust::sequence(rmm::exec_policy(nullptr)->on(nullptr),
                     levels.back().cluster_inv_off,
                     levels.back().cluster_inv_off + levels.back().num_verts + 1);
    thrust::sequence(rmm::exec_policy(nullptr)->on(nullptr),
                     levels.back().cluster_inv_ind,
                     levels.back().cluster_inv_ind + levels.back().num_verts);
  }
  else {
    // assign initialized cluster to cluster_d device vector
    // generate inverse cluster in CSR formate
    if (init_cluster == nullptr) {
      final_modularity = -1;
      return NVLOUVAIN_ERR_BAD_PARAMETERS;
    }

    thrust::copy(init_cluster, init_cluster + n_vertex, levels.back().clusters);
    generate_cluster_inv(levels.back().num_verts,
                         levels.back().num_clusters,
                         levels.back().clusters,
                         levels.back().cluster_inv_off,
                         levels.back().cluster_inv_ind);
  }

  ValType new_Q;
  hr_clock.start();
  compute_k_vec(levels.back().num_verts,
                levels.back().csr_off,
                levels.back().csr_val,
                k_vec_ptr);
  new_Q = compute_modularity(levels.back().num_verts,
                             levels.back().nnz,
                             levels.back().num_clusters,
                             levels.back().csr_off,
                             levels.back().csr_ind,
                             levels.back().csr_val,
                             k_vec_ptr,
                             levels.back().clusters,
                             (ValType) 1.0,
                             m2,
                             e_c_ptr,
                             k_c_ptr,
                             m_c_ptr);

  hr_clock.stop(&timed);
  diff_time = timed;

  LOG() << "Initial modularity value: " << COLOR_MGT << new_Q << COLOR_WHT << " runtime: "
      << diff_time / 1000 << "\n";

  bool contin(true);
//  int bound = 0;
//  int except = 3;

  do {
    m2 = thrust::reduce(rmm::exec_policy(nullptr)->on(nullptr),
                        levels.back().csr_val,
                        levels.back().csr_val + levels.back().nnz);
    LOG() << "Starting do loop: " << levels.back().nnz << " edges, and " << levels.back().num_verts
        << " vertices, m2=" << m2 << " level: " << levels.size() << "\n";

    compute_k_vec((IdxType) levels.back().num_verts,
                  levels.back().csr_off,
                  levels.back().csr_val,
                  k_vec_ptr);

    IdxType num_moved = 1;
    IdxType total_moved = 0;
    IdxType iter_count = 0;
    while (num_moved > 0 && iter_count++ < 10) {
      hr_clock.start();
      ValType new_new_Q = compute_modularity(levels.back().num_verts,
                                             levels.back().nnz,
                                             levels.back().num_clusters,
                                             levels.back().csr_off,
                                             levels.back().csr_ind,
                                             levels.back().csr_val,
                                             k_vec_ptr,
                                             levels.back().clusters,
                                             (ValType) 1.0,
                                             m2,
                                             e_c_ptr,
                                             k_c_ptr,
                                             m_c_ptr);
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
                               delta_Q_arr_ptr,
                               (ValType) 1.0,
                               m2);
      hr_clock.stop(&timed);
      diff_time = timed;
      LOG() << "Complete build_delta_modularity_vector  runtime: " << diff_time / 1000
          << " Current modularity: " << new_new_Q << "\n";

      // Make swaps
      num_moved = makeSwaps(levels.back().num_verts,
                            levels.back().csr_off,
                            levels.back().csr_ind,
                            delta_Q_arr_ptr,
                            levels.back().clusters);

      total_moved += num_moved;

      renumberAndCountAggregates(levels.back().clusters,
                                 levels.back().num_verts,
                                 levels.back().num_clusters);
      LOG() << "Completed makeSwaps: " << num_moved << " swaps made. Now there are "
          << levels.back().num_clusters
          << " clusters\n";

    }

    // If we got through the swapping phase without making any swaps we are done.
    if (total_moved == 0 && levels.size() == 1)
      break;

    if (total_moved == 0) {
      while (levels.size() > 1) {
        // Do some magic and project down the last level to the previous level
        level_info<IdxType, ValType>last = levels.back();
        level_info<IdxType, ValType>prev = levels[levels.size() - 2];
        project(last.num_verts,
                prev.cluster_inv_off,
                prev.cluster_inv_ind,
                last.clusters,
                prev.clusters);
        LOG() << "Projected level " << levels.size() << " down\n";
        // Delete the last frame and pop it off
        levels.back().delete_all();
        levels.pop_back();
      }
      LOG() << "Starting to iterate again on projected result:\n";
      continue;
    }

    // Generate the cluster_inv so the result gets projected if we terminate for max_level
    generate_cluster_inv(levels.back().num_verts,
                         levels.back().num_clusters,
                         levels.back().clusters,
                         levels.back().cluster_inv_off,
                         levels.back().cluster_inv_ind);

    // Check to see if we have computed up to the max level specified:
    if ((IdxType) levels.size() >= max_iter)
      break;

    // If there were swaps made then we add a level on and continue
    level_info<IdxType, ValType>prev = levels.back();
    levels.push_back(level_info<IdxType, ValType>());
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

    LOG() << "Completed generate super vertices graph: num_verts " << levels.back().num_verts
        << " nnz " << levels.back().nnz << "\n";

    IdxType* new_cluster_inv_off;
    IdxType* new_cluster_inv_ind;
    IdxType* new_clusters;
    ALLOC_TRY(&new_cluster_inv_off, sizeof(IdxType) * (prev.num_clusters + 1), nullptr);
    ALLOC_TRY(&new_cluster_inv_ind, sizeof(IdxType) * prev.num_clusters, nullptr);
    ALLOC_TRY(&new_clusters, sizeof(IdxType) * prev.num_clusters, nullptr);
    levels.back().cluster_inv_off = new_cluster_inv_off;
    levels.back().cluster_inv_ind = new_cluster_inv_ind;
    levels.back().clusters = new_clusters;
    thrust::sequence(rmm::exec_policy(nullptr)->on(nullptr),
                     new_clusters,
                     new_clusters + prev.num_clusters,
                     0);
  } while (contin);

  // Now that we have finished with the levels, it's time to project the solution back down
  // As we project the solution down we will also be deleting temporary memory allocations from each level
  // Start off by ignoring the last level, since nothing changed in it.
  if (levels.size() > 1) {
    LOG() << "Popping off level " << levels.size() << "\n";
    levels.back().delete_all();
    levels.pop_back();
  }
  while (levels.size() > 1) {
    // Do some magic and project down the last level to the previous level
    level_info<IdxType, ValType>last = levels.back();
    level_info<IdxType, ValType>prev = levels[levels.size() - 2];
    project(last.num_verts,
            prev.cluster_inv_off,
            prev.cluster_inv_ind,
            last.clusters,
            prev.clusters);
    LOG() << "Projected level " << levels.size() << " down\n";
    // Delete the last frame and pop it off
    levels.back().delete_all();
    levels.pop_back();
  }

  // Compute the final modularity
  renumberAndCountAggregates(levels.back().clusters,
                             levels.back().num_verts,
                             levels.back().num_clusters);
  m2 = thrust::reduce(rmm::exec_policy(nullptr)->on(nullptr),
                      levels.back().csr_val,
                      levels.back().csr_val + levels.back().nnz);
  compute_k_vec((IdxType) levels.back().num_verts,
                levels.back().csr_off,
                levels.back().csr_val,
                k_vec_ptr);
  final_modularity = compute_modularity(levels.back().num_verts,
                                        levels.back().nnz,
                                        levels.back().num_clusters,
                                        levels.back().csr_off,
                                        levels.back().csr_ind,
                                        levels.back().csr_val,
                                        k_vec_ptr,
                                        levels.back().clusters,
                                        (ValType) 1.0,
                                        m2,
                                        e_c_ptr,
                                        k_c_ptr,
                                        m_c_ptr);
  LOG() << "Final modularity " << final_modularity << "\n";

  cudaMemcpy(cluster_vec,
             levels.back().clusters,
             levels.back().num_verts * sizeof(IdxType),
             cudaMemcpyDefault);
  levels.back().delete_added();
  return NVLOUVAIN_OK;
}
} // namespace nvlouvain
