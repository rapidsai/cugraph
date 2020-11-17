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

#include <rmm/thrust_rmm_allocator.h>
#include <utilities/error.hpp>

#include <stdio.h>
#include <converters/COOtoCSR.cuh>
#include <graph.hpp>
#include <internals.hpp>

#include "bh_kernels.hpp"
#include "fa2_kernels.hpp"
#include "utilities/graph_utils.cuh"
#include "utils.hpp"

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
void barnes_hut(GraphCOOView<vertex_t, edge_t, weight_t> &graph,
                float *pos,
                const int max_iter                            = 500,
                float *x_start                                = nullptr,
                float *y_start                                = nullptr,
                bool outbound_attraction_distribution         = true,
                bool lin_log_mode                             = false,
                bool prevent_overlapping                      = false,
                const float edge_weight_influence             = 1.0,
                const float jitter_tolerance                  = 1.0,
                const float theta                             = 0.5,
                const float scaling_ratio                     = 2.0,
                bool strong_gravity_mode                      = false,
                const float gravity                           = 1.0,
                bool verbose                                  = false,
                internals::GraphBasedDimRedCallback *callback = nullptr)
{
  cudaStream_t stream = {nullptr};
  const edge_t e      = graph.number_of_edges;
  const vertex_t n    = graph.number_of_vertices;

  const int blocks = getMultiProcessorCount();
  // A tiny jitter to promote numerical stability/
  const float epssq = 0.0025;
  // We use the same array for nodes and cells.
  int nnodes = n * 2;
  if (nnodes < 1024 * blocks) nnodes = 1024 * blocks;
  while ((nnodes & (32 - 1)) != 0) nnodes++;
  nnodes--;

  // Allocate more space
  //---------------------------------------------------
  rmm::device_vector<unsigned> d_limiter(1);
  rmm::device_vector<int> d_maxdepthd(1);
  rmm::device_vector<int> d_bottomd(1);
  rmm::device_vector<float> d_radiusd(1);

  unsigned *limiter = d_limiter.data().get();
  int *maxdepthd    = d_maxdepthd.data().get();
  int *bottomd      = d_bottomd.data().get();
  float *radiusd    = d_radiusd.data().get();

  InitializationKernel<<<1, 1, 0, stream>>>(limiter, maxdepthd, radiusd);
  CHECK_CUDA(stream);

  const int FOUR_NNODES     = 4 * nnodes;
  const int FOUR_N          = 4 * n;
  const float theta_squared = theta * theta;
  const int NNODES          = nnodes;

  rmm::device_vector<int> d_startl(nnodes + 1, 0);
  rmm::device_vector<int> d_childl((nnodes + 1) * 4, 0);
  // FA2 requires degree + 1
  rmm::device_vector<int> d_massl(nnodes + 1, 1.f);

  rmm::device_vector<float> d_maxxl(blocks * FACTOR1, 0);
  rmm::device_vector<float> d_maxyl(blocks * FACTOR1, 0);
  rmm::device_vector<float> d_minxl(blocks * FACTOR1, 0);
  rmm::device_vector<float> d_minyl(blocks * FACTOR1, 0);

  // Actual mallocs
  int *startl = d_startl.data().get();
  int *childl = d_childl.data().get();
  int *massl  = d_massl.data().get();

  float *maxxl = d_maxxl.data().get();
  float *maxyl = d_maxyl.data().get();
  float *minxl = d_minxl.data().get();
  float *minyl = d_minyl.data().get();

  // SummarizationKernel
  rmm::device_vector<int> d_countl(nnodes + 1, 0);
  int *countl = d_countl.data().get();

  // SortKernel
  rmm::device_vector<int> d_sortl(nnodes + 1, 0);
  int *sortl = d_sortl.data().get();

  // RepulsionKernel
  rmm::device_vector<float> d_rep_forces((nnodes + 1) * 2, 0);
  float *rep_forces = d_rep_forces.data().get();

  rmm::device_vector<float> d_radius_squared(1, 0);
  float *radiusd_squared = d_radius_squared.data().get();

  rmm::device_vector<float> d_nodes_pos((nnodes + 1) * 2, 0);
  float *nodes_pos = d_nodes_pos.data().get();

  // Initialize positions with random values
  int random_state = 0;

  // Copy start x and y positions.
  if (x_start && y_start) {
    copy(n, x_start, nodes_pos);
    copy(n, y_start, nodes_pos + nnodes + 1);
  } else {
    random_vector(nodes_pos, (nnodes + 1) * 2, random_state, stream);
  }

  // Allocate arrays for force computation
  float *attract{nullptr};
  float *old_forces{nullptr};
  float *swinging{nullptr};
  float *traction{nullptr};

  rmm::device_vector<float> d_attract(n * 2, 0);
  rmm::device_vector<float> d_old_forces(n * 2, 0);
  rmm::device_vector<float> d_swinging(n, 0);
  rmm::device_vector<float> d_traction(n, 0);

  attract    = d_attract.data().get();
  old_forces = d_old_forces.data().get();
  swinging   = d_swinging.data().get();
  traction   = d_traction.data().get();

  // Sort COO for coalesced memory access.
  sort(graph, stream);
  CHECK_CUDA(stream);

  graph.degree(massl, cugraph::DegreeDirection::OUT);
  CHECK_CUDA(stream);

  const vertex_t *row = graph.src_indices;
  const vertex_t *col = graph.dst_indices;
  const weight_t *v   = graph.edge_data;

  // Scalars used to adapt global speed.
  float speed                     = 1.f;
  float speed_efficiency          = 1.f;
  float outbound_att_compensation = 1.f;
  float jt                        = 0.f;

  // If outboundAttractionDistribution active, compensate.
  if (outbound_attraction_distribution) {
    int sum =
      thrust::reduce(rmm::exec_policy(stream)->on(stream), d_massl.begin(), d_massl.begin() + n);
    outbound_att_compensation = sum / (float)n;
  }

  //
  // Set cache levels for faster algorithm execution
  //---------------------------------------------------
  cudaFuncSetCacheConfig(BoundingBoxKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(TreeBuildingKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(ClearKernel1, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(ClearKernel2, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(SummarizationKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(SortKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(RepulsionKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(apply_forces_bh, cudaFuncCachePreferL1);

  if (callback) {
    callback->setup<float>(nnodes + 1, 2);
    callback->on_preprocess_end(nodes_pos);
  }

  for (int iter = 0; iter < max_iter; ++iter) {
    // Reset force values
    fill((nnodes + 1) * 2, rep_forces, 0.f);
    fill(n * 2, attract, 0.f);
    fill(n, swinging, 0.f);
    fill(n, traction, 0.f);

    ResetKernel<<<1, 1, 0, stream>>>(radiusd_squared, bottomd, NNODES, radiusd);
    CHECK_CUDA(stream);

    // Compute bounding box arround all bodies
    BoundingBoxKernel<<<blocks * FACTOR1, THREADS1, 0, stream>>>(startl,
                                                                 childl,
                                                                 massl,
                                                                 nodes_pos,
                                                                 nodes_pos + nnodes + 1,
                                                                 maxxl,
                                                                 maxyl,
                                                                 minxl,
                                                                 minyl,
                                                                 FOUR_NNODES,
                                                                 NNODES,
                                                                 n,
                                                                 limiter,
                                                                 radiusd);
    CHECK_CUDA(stream);

    ClearKernel1<<<blocks, 1024, 0, stream>>>(childl, FOUR_NNODES, FOUR_N);
    CHECK_CUDA(stream);

    // Build quadtree
    TreeBuildingKernel<<<blocks * FACTOR2, THREADS2, 0, stream>>>(
      childl, nodes_pos, nodes_pos + nnodes + 1, NNODES, n, maxdepthd, bottomd, radiusd);
    CHECK_CUDA(stream);

    ClearKernel2<<<blocks, 1024, 0, stream>>>(startl, massl, NNODES, bottomd);
    CHECK_CUDA(stream);

    // Summarizes mass and position for each cell, bottom up approach
    SummarizationKernel<<<blocks * FACTOR3, THREADS3, 0, stream>>>(
      countl, childl, massl, nodes_pos, nodes_pos + nnodes + 1, NNODES, n, bottomd);
    CHECK_CUDA(stream);

    // Group closed bodies together, used to speed up Repulsion kernel
    SortKernel<<<blocks * FACTOR4, THREADS4, 0, stream>>>(
      sortl, countl, startl, childl, NNODES, n, bottomd);
    CHECK_CUDA(stream);

    // Force computation O(n . log(n))
    RepulsionKernel<<<blocks * FACTOR5, THREADS5, 0, stream>>>(scaling_ratio,
                                                               theta,
                                                               epssq,
                                                               sortl,
                                                               childl,
                                                               massl,
                                                               nodes_pos,
                                                               nodes_pos + nnodes + 1,
                                                               rep_forces,
                                                               rep_forces + nnodes + 1,
                                                               theta_squared,
                                                               NNODES,
                                                               FOUR_NNODES,
                                                               n,
                                                               radiusd_squared,
                                                               maxdepthd);
    CHECK_CUDA(stream);

    apply_gravity<vertex_t>(nodes_pos,
                            nodes_pos + nnodes + 1,
                            attract,
                            attract + n,
                            massl,
                            gravity,
                            strong_gravity_mode,
                            scaling_ratio,
                            n,
                            stream);

    apply_attraction<vertex_t, edge_t, weight_t>(row,
                                                 col,
                                                 v,
                                                 e,
                                                 nodes_pos,
                                                 nodes_pos + nnodes + 1,
                                                 attract,
                                                 attract + n,
                                                 massl,
                                                 outbound_attraction_distribution,
                                                 lin_log_mode,
                                                 edge_weight_influence,
                                                 outbound_att_compensation,
                                                 stream);

    compute_local_speed(rep_forces,
                        rep_forces + nnodes + 1,
                        attract,
                        attract + n,
                        old_forces,
                        old_forces + n,
                        massl,
                        swinging,
                        traction,
                        n,
                        stream);

    // Compute global swinging and traction values
    const float s =
      thrust::reduce(rmm::exec_policy(stream)->on(stream), d_swinging.begin(), d_swinging.end());

    const float t =
      thrust::reduce(rmm::exec_policy(stream)->on(stream), d_traction.begin(), d_traction.end());

    // Compute global speed based on gloab and local swinging and traction.
    adapt_speed<vertex_t>(jitter_tolerance, &jt, &speed, &speed_efficiency, s, t, n);

    // Update positions
    apply_forces_bh<<<blocks * FACTOR6, THREADS6, 0, stream>>>(nodes_pos,
                                                               nodes_pos + nnodes + 1,
                                                               attract,
                                                               attract + n,
                                                               rep_forces,
                                                               rep_forces + nnodes + 1,
                                                               old_forces,
                                                               old_forces + n,
                                                               swinging,
                                                               speed,
                                                               n);

    if (callback) callback->on_epoch_end(nodes_pos);

    if (verbose) {
      printf("iteration %i, speed: %f, speed_efficiency: %f, ", iter + 1, speed, speed_efficiency);
      printf("jt: %f, ", jt);
      printf("swinging: %f, traction: %f\n", s, t);
    }
  }

  // Copy nodes positions into final output pos
  copy(n, nodes_pos, pos);
  copy(n, nodes_pos + nnodes + 1, pos + n);

  if (callback) callback->on_train_end(nodes_pos);
}

}  // namespace detail
}  // namespace cugraph
