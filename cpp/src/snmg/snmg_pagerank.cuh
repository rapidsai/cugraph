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

// snmg pagerank
// Author: Alex Fender afender@nvidia.com
 
#pragma once
#include "cub/cub.cuh"
#include <omp.h>
#include "graph_utils.cuh"
#include "snmg_utils.cuh"
#include "snmg_spmv.cuh"
//#define SNMG_DEBUG

namespace cugraph
{

    template<typename IndexType, typename ValueType>
  __global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
  transition_kernel(const IndexType e,
                    const IndexType *ind,
                    IndexType *degree,
                    ValueType *val) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < e; i += gridDim.x * blockDim.x)
      val[i] = 1.0 / degree[ind[i]];
  }

  void transition_vals( SNMGinfo & env,
                        const IndexType e,
                        const IndexType *csrInd,
                        const IndexType *degree,
                        ValueType *val) {
    int threads min(e, 256);
    int blocks min(32*env.get_num_sm(), CUDA_MAX_BLOCKS);
    transition_kernel<IndexType, ValueType> <<<blocks, threads>>> (e, csrInd, degree, val);
    cudaCheckError();
  }


} //namespace cugraph
