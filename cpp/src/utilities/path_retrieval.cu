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

#include <utilities/path_retrieval.hpp>
#include "utilities/graph_utils.cuh"
#include <raft/handle.hpp>

namespace cugraph {
  namespace detail {
__global__
  void compute_cost_kernel(int const *vtx_ptr, int const *preds,
    float const *info_weights, float *out, int num_vertices)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
      i < num_vertices;
      i += gridDim.x * blockDim.x) {
    float sum = 0;
    int cur_vtx = vtx_ptr[i];
    int pred = preds[cur_vtx];
    while (pred != -1) {
      sum += info_weights[i];
      pred = preds[pred];
    }
    sum += info_weights[i];
    out[i] = sum;
  }
}
    void get_traversed_path(raft::handle_t const &handle, int const *vtx_ptr,
        int const *preds, float const *info_weights, float *out, int num_vertices)
    {
      dim3 nthreads, nblocks;
      nthreads.x = min(num_vertices, CUDA_MAX_KERNEL_THREADS);
      nthreads.y = 1;
      nthreads.z = 1;
      nblocks.x  = min((num_vertices + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
      nblocks.y  = 1;
      nblocks.z  = 1;

      compute_cost_kernel<<<nblocks, nthreads>>>(vtx_ptr, preds, info_weights,
          out, num_vertices);
    }
  } // namespace detail

  void get_traversed_path(raft::handle_t const &handle, int const *vtx_ptr,
      int const *preds, float const *info_weights, float *out, int num_vertices)
  {
		RAFT_EXPECTS(num_vertices > 0 , "num_vertices should be strictly positive");
		cugraph::detail::get_traversed_path(handle, vtx_ptr, preds, info_weights,
        out, num_vertices);
  }
}  // namespace cugraph
