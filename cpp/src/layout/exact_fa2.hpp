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

#include "exact_repulsion.hpp"
#include "fa2_kernels.hpp"
#include "utils.hpp"

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
void exact_fa2(GraphCOOView<vertex_t, edge_t, weight_t> &graph,
               float *pos,
               const int max_iter                            = 500,
               float *x_start                                = nullptr,
               float *y_start                                = nullptr,
               bool outbound_attraction_distribution         = true,
               bool lin_log_mode                             = false,
               bool prevent_overlapping                      = false,
               const float edge_weight_influence             = 1.0,
               const float jitter_tolerance                  = 1.0,
               const float scaling_ratio                     = 2.0,
               bool strong_gravity_mode                      = false,
               const float gravity                           = 1.0,
               bool verbose                                  = false,
               internals::GraphBasedDimRedCallback *callback = nullptr)
{
  cudaStream_t stream = {nullptr};
  const edge_t e      = graph.number_of_edges;
  const vertex_t n    = graph.number_of_vertices;

  float *d_repel{nullptr};
  float *d_attract{nullptr};
  float *d_old_forces{nullptr};
  int *d_mass{nullptr};
  float *d_swinging{nullptr};
  float *d_traction{nullptr};

  rmm::device_vector<float> repel(n * 2, 0);
  rmm::device_vector<float> attract(n * 2, 0);
  rmm::device_vector<float> old_forces(n * 2, 0);
  // FA2 requires degree + 1.
  rmm::device_vector<int> mass(n, 1);
  rmm::device_vector<float> swinging(n, 0);
  rmm::device_vector<float> traction(n, 0);

  d_repel      = repel.data().get();
  d_attract    = attract.data().get();
  d_old_forces = old_forces.data().get();
  d_mass       = mass.data().get();
  d_swinging   = swinging.data().get();
  d_traction   = traction.data().get();

  int random_state = 0;
  random_vector(pos, n * 2, random_state, stream);

  if (x_start && y_start) {
    copy(n, x_start, pos);
    copy(n, y_start, pos + n);
  }

  // Sort COO for coalesced memory access.
  sort(graph, stream);
  CHECK_CUDA(stream);

  graph.degree(d_mass, cugraph::DegreeDirection::OUT);
  CHECK_CUDA(stream);

  const vertex_t *row = graph.src_indices;
  const vertex_t *col = graph.dst_indices;
  const weight_t *v   = graph.edge_data;

  float speed                     = 1.f;
  float speed_efficiency          = 1.f;
  float outbound_att_compensation = 1.f;
  float jt                        = 0.f;

  if (outbound_attraction_distribution) {
    int sum = thrust::reduce(rmm::exec_policy(stream)->on(stream), mass.begin(), mass.end());
    outbound_att_compensation = sum / (float)n;
  }

  if (callback) {
    callback->setup<float>(n, 2);
    callback->on_preprocess_end(pos);
  }

  for (int iter = 0; iter < max_iter; ++iter) {
    // Reset force arrays
    fill(n * 2, d_repel, 0.f);
    fill(n * 2, d_attract, 0.f);
    fill(n, d_swinging, 0.f);
    fill(n, d_traction, 0.f);

    // Exact repulsion
    apply_repulsion<vertex_t>(pos, pos + n, d_repel, d_repel + n, d_mass, scaling_ratio, n, stream);

    apply_gravity<vertex_t>(pos,
                            pos + n,
                            d_attract,
                            d_attract + n,
                            d_mass,
                            gravity,
                            strong_gravity_mode,
                            scaling_ratio,
                            n,
                            stream);

    apply_attraction<vertex_t, edge_t, weight_t>(row,
                                                 col,
                                                 v,
                                                 e,
                                                 pos,
                                                 pos + n,
                                                 d_attract,
                                                 d_attract + n,
                                                 d_mass,
                                                 outbound_attraction_distribution,
                                                 lin_log_mode,
                                                 edge_weight_influence,
                                                 outbound_att_compensation,
                                                 stream);

    compute_local_speed(d_repel,
                        d_repel + n,
                        d_attract,
                        d_attract + n,
                        d_old_forces,
                        d_old_forces + n,
                        d_mass,
                        d_swinging,
                        d_traction,
                        n,
                        stream);

    // Compute global swinging and traction values.
    const float s =
      thrust::reduce(rmm::exec_policy(stream)->on(stream), swinging.begin(), swinging.end());
    const float t =
      thrust::reduce(rmm::exec_policy(stream)->on(stream), traction.begin(), traction.end());

    adapt_speed<vertex_t>(jitter_tolerance, &jt, &speed, &speed_efficiency, s, t, n);

    apply_forces<vertex_t>(pos,
                           pos + n,
                           d_repel,
                           d_repel + n,
                           d_attract,
                           d_attract + n,
                           d_old_forces,
                           d_old_forces + n,
                           d_swinging,
                           speed,
                           n,
                           stream);

    if (callback) callback->on_epoch_end(pos);

    if (verbose) {
      printf("iteration %i, speed: %f, speed_efficiency: %f, ", iter + 1, speed, speed_efficiency);
      printf("jt: %f, ", jt);
      printf("swinging: %f, traction: %f\n", s, t);
    }
  }

  if (callback) callback->on_train_end(pos);
}

}  // namespace detail
}  // namespace cugraph
