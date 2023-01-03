/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <raft/core/handle.hpp>

#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/path_retrieval.hpp>

#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename weight_t>
__global__ void get_traversed_cost_kernel(vertex_t const* vertices,
                                          vertex_t const* preds,
                                          vertex_t const* vtx_map,
                                          weight_t const* info_weights,
                                          weight_t* out,
                                          vertex_t stop_vertex,
                                          vertex_t num_vertices)
{
  for (vertex_t i = threadIdx.x + blockIdx.x * blockDim.x; i < num_vertices;
       i += gridDim.x * blockDim.x) {
    weight_t sum  = info_weights[i];
    vertex_t pred = preds[i];
    while (pred != stop_vertex) {
      vertex_t pos = vtx_map[pred];
      sum += info_weights[pos];
      pred = preds[pos];
    }
    out[i] = sum;
  }
}

template <typename vertex_t, typename weight_t>
void get_traversed_cost_impl(raft::handle_t const& handle,
                             vertex_t const* vertices,
                             vertex_t const* preds,
                             weight_t const* info_weights,
                             weight_t* out,
                             vertex_t stop_vertex,
                             vertex_t num_vertices)
{
  auto stream          = handle.get_stream();
  vertex_t max_blocks  = handle.get_device_properties().maxGridSize[0];
  vertex_t max_threads = handle.get_device_properties().maxThreadsPerBlock;

  dim3 nthreads, nblocks;
  nthreads.x = std::min<vertex_t>(num_vertices, max_threads);
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x  = std::min<vertex_t>((num_vertices + nthreads.x - 1) / nthreads.x, max_blocks);
  nblocks.y  = 1;
  nblocks.z  = 1;

  rmm::device_uvector<vertex_t> vtx_map_v(num_vertices, stream);
  rmm::device_uvector<vertex_t> vtx_keys_v(num_vertices, stream);
  vertex_t* vtx_map  = vtx_map_v.data();
  vertex_t* vtx_keys = vtx_keys_v.data();
  raft::copy(vtx_keys, vertices, num_vertices, stream);

  thrust::sequence(handle.get_thrust_policy(), vtx_map, vtx_map + num_vertices);

  thrust::stable_sort_by_key(
    handle.get_thrust_policy(), vtx_keys, vtx_keys + num_vertices, vtx_map);

  get_traversed_cost_kernel<<<nblocks, nthreads>>>(
    vertices, preds, vtx_map, info_weights, out, stop_vertex, num_vertices);
}
}  // namespace detail

template <typename vertex_t, typename weight_t>
void get_traversed_cost(raft::handle_t const& handle,
                        vertex_t const* vertices,
                        vertex_t const* preds,
                        weight_t const* info_weights,
                        weight_t* out,
                        vertex_t stop_vertex,
                        vertex_t num_vertices)
{
  CUGRAPH_EXPECTS(num_vertices > 0, "num_vertices should be strictly positive");
  CUGRAPH_EXPECTS(out != nullptr, "out should be of size num_vertices");
  cugraph::detail::get_traversed_cost_impl(
    handle, vertices, preds, info_weights, out, stop_vertex, num_vertices);
}

template void get_traversed_cost<int32_t, float>(raft::handle_t const& handle,
                                                 int32_t const* vertices,
                                                 int32_t const* preds,
                                                 float const* info_weights,
                                                 float* out,
                                                 int32_t stop_vertex,
                                                 int32_t num_vertices);

template void get_traversed_cost<int32_t, double>(raft::handle_t const& handle,
                                                  int32_t const* vertices,
                                                  int32_t const* preds,
                                                  double const* info_weights,
                                                  double* out,
                                                  int32_t stop_vertex,
                                                  int32_t num_vertices);

template void get_traversed_cost<int64_t, float>(raft::handle_t const& handle,
                                                 int64_t const* vertices,
                                                 int64_t const* preds,
                                                 float const* info_weights,
                                                 float* out,
                                                 int64_t stop_vertex,
                                                 int64_t num_vertices);

template void get_traversed_cost<int64_t, double>(raft::handle_t const& handle,
                                                  int64_t const* vertices,
                                                  int64_t const* preds,
                                                  double const* info_weights,
                                                  double* out,
                                                  int64_t stop_vertex,
                                                  int64_t num_vertices);
}  // namespace cugraph
