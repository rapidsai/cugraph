/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <converters/legacy/COOtoCSR.cuh>

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/legacy/graph.hpp>
#include <cugraph/legacy/internals.hpp>
#include <cugraph/utilities/error.hpp>

#include <thrust/fill.h>
#include <thrust/reduce.h>

#include "exact_repulsion.cuh"
#include "fa2_kernels.cuh"
#include "utils.hpp"

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
void exact_fa2(raft::handle_t const& handle,
               legacy::GraphCOOView<vertex_t, edge_t, weight_t>& graph,
               float* pos,
               const int max_iter                            = 500,
               float* x_start                                = nullptr,
               float* y_start                                = nullptr,
               bool outbound_attraction_distribution         = true,
               bool lin_log_mode                             = false,
               bool prevent_overlapping                      = false,
               const float edge_weight_influence             = 1.0,
               const float jitter_tolerance                  = 1.0,
               const float scaling_ratio                     = 2.0,
               bool strong_gravity_mode                      = false,
               const float gravity                           = 1.0,
               bool verbose                                  = false,
               internals::GraphBasedDimRedCallback* callback = nullptr)
{
  auto stream_view = handle.get_stream();
  const edge_t e   = graph.number_of_edges;
  const vertex_t n = graph.number_of_vertices;

  float* d_repel{nullptr};
  float* d_attract{nullptr};
  float* d_old_forces{nullptr};
  int* d_mass{nullptr};
  float* d_swinging{nullptr};
  float* d_traction{nullptr};

  rmm::device_uvector<float> repel(n * 2, stream_view);
  rmm::device_uvector<float> attract(n * 2, stream_view);
  rmm::device_uvector<float> old_forces(n * 2, stream_view);
  thrust::fill(handle.get_thrust_policy(), old_forces.begin(), old_forces.end(), 0.f);
  // FA2 requires degree + 1.
  rmm::device_uvector<int> mass(n, stream_view);
  thrust::fill(handle.get_thrust_policy(), mass.begin(), mass.end(), 1);
  rmm::device_uvector<float> swinging(n, stream_view);
  rmm::device_uvector<float> traction(n, stream_view);

  d_repel      = repel.data();
  d_attract    = attract.data();
  d_old_forces = old_forces.data();
  d_mass       = mass.data();
  d_swinging   = swinging.data();
  d_traction   = traction.data();

  uniform_random_fill(handle.get_stream(), pos, n * 2, -100.0f, 100.0f, uint64_t{0});

  if (x_start && y_start) {
    raft::copy(pos, x_start, n, stream_view.value());
    raft::copy(pos + n, y_start, n, stream_view.value());
  }

  // Sort COO for coalesced memory access.
  sort(graph, stream_view.value());
  RAFT_CHECK_CUDA(stream_view.value());

  graph.degree(d_mass, cugraph::legacy::DegreeDirection::OUT);
  RAFT_CHECK_CUDA(stream_view.value());

  const vertex_t* row = graph.src_indices;
  const vertex_t* col = graph.dst_indices;
  const weight_t* v   = graph.edge_data;

  float speed                     = 1.f;
  float speed_efficiency          = 1.f;
  float outbound_att_compensation = 1.f;
  float jt                        = 0.f;

  if (outbound_attraction_distribution) {
    int sum = thrust::reduce(handle.get_thrust_policy(), mass.begin(), mass.end());
    outbound_att_compensation = sum / (float)n;
  }

  if (callback) {
    callback->setup<float>(n, 2);
    callback->on_preprocess_end(pos);
  }

  for (int iter = 0; iter < max_iter; ++iter) {
    // Reset force arrays
    thrust::fill(handle.get_thrust_policy(), repel.begin(), repel.end(), 0.f);
    thrust::fill(handle.get_thrust_policy(), attract.begin(), attract.end(), 0.f);
    thrust::fill(handle.get_thrust_policy(), swinging.begin(), swinging.end(), 0.f);
    thrust::fill(handle.get_thrust_policy(), traction.begin(), traction.end(), 0.f);

    // Exact repulsion
    apply_repulsion<vertex_t>(
      pos, pos + n, d_repel, d_repel + n, d_mass, scaling_ratio, n, stream_view.value());

    apply_gravity<vertex_t>(pos,
                            pos + n,
                            d_attract,
                            d_attract + n,
                            d_mass,
                            gravity,
                            strong_gravity_mode,
                            scaling_ratio,
                            n,
                            stream_view.value());

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
                                                 stream_view.value());

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
                        stream_view.value());

    // Compute global swinging and traction values.
    const float s = thrust::reduce(handle.get_thrust_policy(), swinging.begin(), swinging.end());
    const float t = thrust::reduce(handle.get_thrust_policy(), traction.begin(), traction.end());

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
                           stream_view.value());

    if (callback) callback->on_epoch_end(pos);

    if (verbose) {
      std::cout << "iteration: " << iter + 1 << ", speed: " << speed
                << ", speed_efficiency: " << speed_efficiency << ", jt: " << jt
                << ", swinging: " << s << ", traction: " << t << "\n";
    }
  }

  if (callback) callback->on_train_end(pos);
}

}  // namespace detail
}  // namespace cugraph
