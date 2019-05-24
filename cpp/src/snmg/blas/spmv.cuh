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

// snmg spmv
// Author: Alex Fender afender@nvidia.com
 
#pragma once
#include "cub/cub.cuh"
#include <omp.h>
#include "utilities/graph_utils.cuh"
#include "snmg/utils.cuh"
//#define SNMG_DEBUG

namespace cugraph
{

template <typename IndexType, typename ValueType>
class SNMGcsrmv 
{ 

  private:
    size_t v_glob;
    size_t v_loc;
    size_t e_loc;
    SNMGinfo env;
    size_t* part_off;
    int i;
    int p;
    IndexType * off;
    IndexType * ind;
    ValueType * val;
    ValueType * y_loc;
    cudaStream_t stream;
    void* cub_d_temp_storage;
    size_t cub_temp_storage_bytes;

  public: 
    SNMGcsrmv(SNMGinfo & env_, size_t* part_off_, 
              IndexType * off_, IndexType * ind_, ValueType * val_, ValueType ** x) : 
              env(env_), part_off(part_off_), off(off_), ind(ind_), val(val_) { 
      sync_all();
      cub_d_temp_storage = NULL;
      cub_temp_storage_bytes = 0;
      stream = nullptr;
      i = env.get_thread_num();
      p = env.get_num_threads(); 
      v_glob = part_off[p];
      v_loc = part_off[i+1]-part_off[i];
      IndexType tmp;
      cudaMemcpy(&tmp, &off[v_loc], sizeof(IndexType),cudaMemcpyDeviceToHost);
      cudaCheckError();
      e_loc = tmp;

      // Allocate the local result
      ALLOC_TRY ((void**)&y_loc, v_loc*sizeof(ValueType), stream);

      // get temporary storage size for CUB
      cub::DeviceSpmv::CsrMV(cub_d_temp_storage, cub_temp_storage_bytes, 
                                      val, off, ind, x[i], y_loc, v_loc, v_glob, e_loc);
      cudaCheckError();
      // Allocate CUB's temporary storage
      ALLOC_TRY ((void**)&cub_d_temp_storage, cub_temp_storage_bytes, stream);
    } 

    ~SNMGcsrmv() { 
      ALLOC_FREE_TRY(cub_d_temp_storage, stream);
      ALLOC_FREE_TRY(y_loc, stream);
    }

    // run the power iteration
    void run (ValueType ** x) {
    // Local SPMV
    cub::DeviceSpmv::CsrMV(cub_d_temp_storage, cub_temp_storage_bytes, 
                                    val, off, ind, x[i], y_loc, v_loc, v_glob, e_loc);
    cudaCheckError()
    sync_all();
     
 #ifdef SNMG_DEBUG
    print_mem_usage();  
    #pragma omp master 
    {std::cout <<  omp_get_wtime() - t << " ";}
     Wait for all local spmv
    t = omp_get_wtime();
    sync_all();
    #pragma omp master 
    {std::cout <<  omp_get_wtime() - t << " ";}
    Update the output vector
#endif
     
    allgather (env, part_off, y_loc, x);
  }
};


} //namespace cugraph
