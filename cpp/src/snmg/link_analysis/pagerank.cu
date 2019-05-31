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
 
#include "cub/cub.cuh"
#include <omp.h>
#include "rmm_utils.h"
#include "utilities/graph_utils.cuh"
#include "snmg/utils.cuh"
#include "utilities/cusparse_helper.h"
#include "snmg/blas/spmv.cuh"
#include "snmg/link_analysis/pagerank.cuh"
#include "snmg/degree/degree.cuh"
//#define SNMG_DEBUG

namespace cugraph
{

  template<typename IndexType, typename ValueType>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
transition_kernel(const size_t e,
                  const IndexType *ind,
                  const IndexType *degree,
                  ValueType *val) {
  for (auto i = threadIdx.x + blockIdx.x * blockDim.x; 
       i < e; 
       i += gridDim.x * blockDim.x)
    val[i] = 1.0 / degree[ind[i]];
}

template <typename IndexType, typename ValueType>
SNMGpagerank<IndexType,ValueType>::SNMGpagerank(SNMGinfo & env_, size_t* part_off_, 
             IndexType * off_, IndexType * ind_) : 
             env(env_), part_off(part_off_), off(off_), ind(ind_) { 
  id = env.get_thread_num();
  nt = env.get_num_threads(); 
  v_glob = part_off[nt];
  v_loc = part_off[id+1]-part_off[id];
  IndexType tmp_e;
  cudaMemcpy(&tmp_e, &off[v_loc], sizeof(IndexType),cudaMemcpyDeviceToHost);
  cudaCheckError();
  e_loc = tmp_e;
  stream = nullptr;
  is_setup = false;
  ALLOC_TRY ((void**)&bookmark,   sizeof(ValueType) * v_glob, stream);
  ALLOC_TRY ((void**)&val, sizeof(ValueType) * e_loc, stream);

  // intialize cusparse. This can take some time.
  Cusparse::get_handle();
} 

template <typename IndexType, typename ValueType>
SNMGpagerank<IndexType,ValueType>::~SNMGpagerank() { 
  Cusparse::destroy_handle();
  ALLOC_FREE_TRY(bookmark, stream); 
  ALLOC_FREE_TRY(val, stream);
}

template <typename IndexType, typename ValueType>
void SNMGpagerank<IndexType,ValueType>::transition_vals(const IndexType *degree) {
  int threads = min(static_cast<IndexType>(e_loc), 256);
  int blocks = min(static_cast<IndexType>(32*env.get_num_sm()), CUDA_MAX_BLOCKS);
  transition_kernel<IndexType, ValueType> <<<blocks, threads>>> (e_loc, ind, degree, val);
  cudaCheckError();
}

template <typename IndexType, typename ValueType>
void SNMGpagerank<IndexType,ValueType>::flag_leafs(const IndexType *degree) {
  int threads = min(static_cast<IndexType>(v_glob), 256);
  int blocks = min(static_cast<IndexType>(32*env.get_num_sm()), CUDA_MAX_BLOCKS);
  flag_leafs_kernel<IndexType, ValueType> <<<blocks, threads>>> (v_glob, degree, bookmark);
  cudaCheckError();
}    


// Artificially create the google matrix by setting val and bookmark
template <typename IndexType, typename ValueType>
void SNMGpagerank<IndexType,ValueType>::setup(ValueType _alpha, IndexType** degree) {
  if (!is_setup) {

    alpha=_alpha;
    ValueType zero = 0.0; 
    IndexType *degree_loc;
    ALLOC_TRY ((void**)&degree_loc,   sizeof(IndexType) * v_glob, stream);
    degree[id] = degree_loc;
    if (snmg_degree(1, part_off, off, ind, degree))
       throw std::string("SNMG Degree failed in Pagerank");

    // Update dangling node vector
    fill(v_glob, bookmark, zero);
    flag_leafs(degree_loc);
    update_dangling_nodes(v_glob, bookmark, alpha);

    // Transition matrix
    transition_vals(degree_loc);

    //exit
    ALLOC_FREE_TRY(degree_loc, stream);
    is_setup = true;
  }
  else
    throw std::string("Setup can be called only once");
}

// run the power iteration on the google matrix
template <typename IndexType, typename ValueType>
void SNMGpagerank<IndexType,ValueType>::solve (int max_iter, ValueType ** pagerank) {
  if (is_setup) {
    ValueType  dot_res;
    ValueType one = 1.0;
    ValueType *pr = pagerank[id];
    fill(v_glob, pagerank[id], one/v_glob);
    dot_res = dot( v_glob, bookmark, pr);
    SNMGcsrmv<IndexType,ValueType> spmv_solver(env, part_off, off, ind, val, pagerank);
    for (auto i = 0; i < max_iter; ++i) {
      spmv_solver.run(pagerank);
      scal(v_glob, alpha, pr);
      addv(v_glob, dot_res * (one/v_glob) , pr);
      dot_res = dot( v_glob, bookmark, pr);
      scal(v_glob, one/nrm2(v_glob, pr) , pr);
    }
    scal(v_glob, one/nrm1(v_glob,pr), pr);
  }
  else {
      throw std::string("Solve was called before setup");
  }
}

template class SNMGpagerank<int, double>;
template class SNMGpagerank<int, float>;


} //namespace cugraph
