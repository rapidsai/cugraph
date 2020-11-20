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

#include <raft/cudart_utils.h>

#include <thrust/random.h>

namespace cugraph {
namespace detail {

struct prg {
  __host__ __device__ float operator()(int n)
  {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(-100.f, 100.f);
    rng.discard(n);
    return dist(rng);
  }
};

void random_vector(float *vec, int n, int seed, cudaStream_t stream)
{
  thrust::counting_iterator<uint32_t> index(seed);
  thrust::transform(rmm::exec_policy(stream)->on(stream), index, index + n, vec, prg());
}

/** helper method to get multi-processor count parameter */
inline int getMultiProcessorCount()
{
  int devId;
  CUDA_TRY(cudaGetDevice(&devId));
  int mpCount;
  CUDA_TRY(cudaDeviceGetAttribute(&mpCount, cudaDevAttrMultiProcessorCount, devId));
  return mpCount;
}

}  // namespace detail
}  // namespace cugraph
