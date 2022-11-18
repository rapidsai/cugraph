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

namespace cugraph {
namespace detail {
// Dependecy Accumulation: based on McLaughlin and Bader, 2018
// FIXME: Accumulation kernel mights not scale well, as each thread is handling
//        all the edges for each node, an approach similar to the traversal
//        bucket (i.e. BFS / SSSP) system might enable speed up.
//        Should look into forAllEdge type primitive for different
//        load balancing
template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
__global__ void edges_accumulation_kernel(result_t* betweenness,
                                          vertex_t number_vertices,
                                          vertex_t const* indices,
                                          edge_t const* offsets,
                                          vertex_t* distances,
                                          double* sp_counters,
                                          double* deltas,
                                          vertex_t depth)
{
  for (int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; thread_idx < number_vertices;
       thread_idx += gridDim.x * blockDim.x) {
    vertex_t vertex     = thread_idx;
    double vertex_delta = 0;
    double vertex_sigma = sp_counters[vertex];
    if (distances[vertex] == depth) {
      edge_t first_edge_idx = offsets[vertex];
      edge_t last_edge_idx  = offsets[vertex + 1];
      for (edge_t edge_idx = first_edge_idx; edge_idx < last_edge_idx; ++edge_idx) {
        vertex_t successor = indices[edge_idx];
        if (distances[successor] == distances[vertex] + 1) {
          double factor = (static_cast<double>(1) + deltas[successor]) / sp_counters[successor];
          double coefficient = vertex_sigma * factor;

          vertex_delta += coefficient;
          betweenness[edge_idx] += coefficient;
        }
      }
      deltas[vertex] = vertex_delta;
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
__global__ void endpoints_accumulation_kernel(result_t* betweenness,
                                              vertex_t number_vertices,
                                              vertex_t const* indices,
                                              edge_t const* offsets,
                                              vertex_t* distances,
                                              double* sp_counters,
                                              double* deltas,
                                              vertex_t depth)
{
  for (int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; thread_idx < number_vertices;
       thread_idx += gridDim.x * blockDim.x) {
    vertex_t vertex     = thread_idx;
    double vertex_delta = 0;
    double vertex_sigma = sp_counters[vertex];
    if (distances[vertex] == depth) {
      edge_t first_edge_idx = offsets[vertex];
      edge_t last_edge_idx  = offsets[vertex + 1];
      for (edge_t edge_idx = first_edge_idx; edge_idx < last_edge_idx; ++edge_idx) {
        vertex_t successor = indices[edge_idx];
        if (distances[successor] == distances[vertex] + 1) {
          double factor = (static_cast<double>(1) + deltas[successor]) / sp_counters[successor];
          vertex_delta += vertex_sigma * factor;
        }
      }
      betweenness[vertex] += 1;
      deltas[vertex] = vertex_delta;
    }
  }
}
template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
__global__ void accumulation_kernel(result_t* betweenness,
                                    vertex_t number_vertices,
                                    vertex_t const* indices,
                                    edge_t const* offsets,
                                    vertex_t* distances,
                                    double* sp_counters,
                                    double* deltas,
                                    vertex_t depth)
{
  for (int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; thread_idx < number_vertices;
       thread_idx += gridDim.x * blockDim.x) {
    vertex_t vertex     = thread_idx;
    double vertex_delta = 0;
    double vertex_sigma = sp_counters[vertex];
    if (distances[vertex] == depth) {
      edge_t first_edge_idx = offsets[vertex];
      edge_t last_edge_idx  = offsets[vertex + 1];
      for (edge_t edge_idx = first_edge_idx; edge_idx < last_edge_idx; ++edge_idx) {
        vertex_t successor = indices[edge_idx];
        if (distances[successor] == distances[vertex] + 1) {
          double factor = (static_cast<double>(1) + deltas[successor]) / sp_counters[successor];
          vertex_delta += vertex_sigma * factor;
        }
      }
      deltas[vertex] = vertex_delta;
    }
  }
}
}  // namespace detail
}  // namespace cugraph
