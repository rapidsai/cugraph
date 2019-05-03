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
#include "snmg/utils.cuh"
#include "snmg/spmv.cuh"
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

template <typename IndexType, typename ValueType>
class SNMGpagerank 
{ 
  private:
    size_t v_glob;
    size_t v_loc;
    size_t e_loc;
    SNMGinfo env;
    size_t* part_off;
    IndexType * off;
    IndexType * ind;
    ValueType * val;
    ValueType * a;
    ValueType * b;
    ValueType * tmp;
    ValueType alpha;
    bool converged;
    bool is_setup;
    cudaStream_t stream;
    int id;
    int np;

  public: 
    SNMGpagerank(SNMGinfo & env_, size_t* part_off_, 
                 IndexType * off_, IndexType * ind_) : 
                 env(env_), part_off(part_off_), off(off_), ind(ind_) { 
      id = env.get_thread_num();
      np = env.get_num_threads(); 
      v_glob = part_off[np];
      v_loc = part_off[id+1]-part_off[id];
      IndexType tmp_e;
      cudaMemcpy(&tmp_e, &off[v_loc], sizeof(IndexType),cudaMemcpyDeviceToHost);
      cudaCheckError();
      e_loc = tmp_e;
      stream = nullptr;
      ALLOC_MANAGED_TRY ((void**)&a,   sizeof(ValueType) * v_glob, stream);
      ALLOC_MANAGED_TRY ((void**)&b,   sizeof(ValueType) * v_glob, stream);
      ALLOC_MANAGED_TRY ((void**)&tmp, sizeof(ValueType) * v_glob, stream);
      ALLOC_MANAGED_TRY ((void**)&val, sizeof(ValueType) * e_loc, stream);
    } 
    ~SNMGpagerank() { 
      ALLOC_FREE_TRY(a, stream); 
      ALLOC_FREE_TRY(b, stream);  
      ALLOC_FREE_TRY(tmp, stream);
      ALLOC_FREE_TRY(val, stream);
    }

    void transition_vals(const IndexType *degree) {
      int threads = min(static_cast<IndexType>(e_loc), 256);
      int blocks = min(static_cast<IndexType>(32*env.get_num_sm()), CUDA_MAX_BLOCKS);
      transition_kernel<IndexType, ValueType> <<<blocks, threads>>> (e_loc, off, degree, val);
      cudaCheckError();
    }

    // compute degree and tansition matrix 
    // set _val, _a, _b, tmp.
    bool setup(ValueType _alpha) {
      alpha=_alpha;
      ValueType randomProbability =  static_cast<ValueType>( 1.0/v_glob);
      fill(v_glob, tmp, randomProbability);
      fill(v_glob, b, randomProbability);
      
      // TODO degree
      
      //transition_vals(degree);
      is_setup=true;
      return true;
    }

    // run the power iteration
    bool solve (float tolerance, int max_iter, ValueType ** pagerank) {
    converged = false;
    ValueType  dot_res;
    ValueType residual;
    ValueType pr = pagerank[id];
    int iter;
    fill(v_glob, pagerank[id], static_cast<ValueType>( 1.0/v_glob));
    SNMGcsrmv<IndexType,ValueType> spmv_solver(env, part_off, off, ind, val, pagerank);
    for (iter = 0; iter < max_iter; ++iter) {
      spmv_solver.run(pagerank);
      scal(v_glob, alpha, pr);
      dot_res = dot( v_glob, a, tmp);
      axpy(v_glob, dot_res,  b,  pr);
      scal(v_glob, (ValueType)1.0/nrm2(v_glob, pr) , pr);
      axpy(v_glob, (ValueType)-1.0,  pr,  tmp);
      residual = nrm2(v_glob, tmp);
      if (residual < tolerance) {
          scal(v_glob, (ValueType)1.0/nrm1(v_glob,pr), pr);
          converged = true;
          break;
      }
      else {
          if (iter< max_iter) {
              std::swap(pr, tmp);
          }
          else {
             scal(v_glob, (ValueType)1.0/nrm1(v_glob,pr), pr);
          }
      }
    }
  }
};

} //namespace cugraph
