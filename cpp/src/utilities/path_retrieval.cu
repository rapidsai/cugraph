/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <raft/handle.hpp>

#include <utilities/path_retrieval.hpp>
#include "utilities/graph_utils.cuh"

namespace cugraph {
namespace detail {

template <typename weight_t>
__global__ void get_traversed_cost_kernel(int const *vertices,
                                          int const *preds,
                                          int const *vtx_map,
                                          weight_t const *info_weights,
                                          weight_t *out,
                                          int num_vertices)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_vertices;
       i += gridDim.x * blockDim.x) {
    float sum = info_weights[i];
    int pred  = preds[i];
    while (pred != -1) {
      int pos = vtx_map[pred];
      sum += info_weights[pos];
      pred = preds[pos];
    }
    out[i] = sum;
  }
}

template <typename weight_t>
void get_traversed_cost_impl(raft::handle_t const &handle,
                             int const *vertices,
                             int const *preds,
                             weight_t const *info_weights,
                             weight_t *out,
                             int num_vertices)
{
  dim3 nthreads, nblocks;
  nthreads.x = min(num_vertices, CUDA_MAX_KERNEL_THREADS);
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x  = min((num_vertices + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
  nblocks.y  = 1;
  nblocks.z  = 1;

  auto stream = handle.get_stream();
  rmm::device_uvector<int> vtx_map_v(num_vertices, stream);
  rmm::device_uvector<int> vtx_keys_v(num_vertices, stream);
  auto *vtx_map  = vtx_map_v.data();
  auto *vtx_keys = vtx_keys_v.data();
  raft::copy(vtx_keys, vertices, num_vertices, stream);

  thrust::sequence(rmm::exec_policy(stream)->on(stream), vtx_map, vtx_map + num_vertices);

  thrust::stable_sort_by_key(
    rmm::exec_policy(stream)->on(stream), vtx_keys, vtx_keys + num_vertices, vtx_map);

  /*
  raft::print_device_vector("vertices: ", vertices, num_vertices, std::cout);
  raft::print_device_vector("vtx_keys: ", vtx_keys, num_vertices, std::cout);
  raft::print_device_vector("preds: ", preds, num_vertices, std::cout);
  raft::print_device_vector("info_weights: ", info_weights, num_vertices, std::cout);
  raft::print_device_vector("vtx_map: ", vtx_map, num_vertices, std::cout);
  */

  get_traversed_cost_kernel<<<nblocks, nthreads>>>(
    vertices, preds, vtx_map, info_weights, out, num_vertices);
  // raft::print_device_vector("out: ", out, num_vertices, std::cout);
}
}  // namespace detail

template <typename weight_t>
void get_traversed_cost(raft::handle_t const &handle,
                        int const *vertices,
                        int const *preds,
                        weight_t const *info_weights,
                        weight_t *out,
                        int num_vertices)
{
  RAFT_EXPECTS(num_vertices > 0, "num_vertices should be strictly positive");
  cugraph::detail::get_traversed_cost_impl(
    handle, vertices, preds, info_weights, out, num_vertices);
}

template void get_traversed_cost<float>(raft::handle_t const &handle,
                                        int const *vertices,
                                        int const *preds,
                                        float const *info_weights,
                                        float *out,
                                        int num_vertices);
template void get_traversed_cost<double>(raft::handle_t const &handle,
                                         int const *vertices,
                                         int const *preds,
                                         double const *info_weights,
                                         double *out,
                                         int num_vertices);
}  // namespace cugraph
