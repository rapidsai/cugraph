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
#include "rmm_utils.h"
#include "utilities/cusparse_helper.h"
#include "spmv.cuh"


namespace cugraph { 
namespace snmg {
template <typename IndexType, typename ValueType>
SNMGcsrmv<IndexType,ValueType>::SNMGcsrmv(SNMGinfo & env_, size_t* part_off_, 
          IndexType * off_, IndexType * ind_, ValueType * val_, ValueType ** x): 
          env(env_), part_off(part_off_), off(off_), ind(ind_), val(val_) { 
  sync_all();
  stream = nullptr;
  i = env.get_thread_num();
  p = env.get_num_threads(); 
  v_glob = part_off[p];
  v_loc = part_off[i+1]-part_off[i];
  IndexType tmp;
  cudaMemcpy(&tmp, &off[v_loc], sizeof(IndexType),cudaMemcpyDeviceToHost);
  CUDA_CHECK_LAST();
  e_loc = tmp;

  // Allocate the local result
  ALLOC_TRY ((void**)&y_loc, v_loc*sizeof(ValueType), stream);

  ValueType h_one = 1.0;
  ValueType h_zero = 0.0;
  spmv.setup(v_loc, v_glob, e_loc, &h_one, val, off, ind, x[i], &h_zero, y_loc);
} 

template <typename IndexType, typename ValueType>
SNMGcsrmv<IndexType,ValueType>::~SNMGcsrmv() { 
  ALLOC_FREE_TRY(y_loc, stream);
}

template <typename IndexType, typename ValueType>
void SNMGcsrmv<IndexType,ValueType>::run (ValueType ** x) {
  sync_all();
  ValueType h_one = 1.0;
  ValueType h_zero = 0.0;
  spmv.run(v_loc, v_glob, e_loc, &h_one, val, off, ind, x[i], &h_zero, y_loc);

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
  sync_all();
  allgather (env, part_off, y_loc, x);
}

template class SNMGcsrmv<int, double>;
template class SNMGcsrmv<int, float>;

template <typename idx_t,typename val_t>
void snmg_csrmv_impl (size_t * part_offsets, gdf_column * off, gdf_column * ind, gdf_column * val, gdf_column ** x_cols){
  
  CUGRAPH_EXPECTS( part_offsets != nullptr, "Invalid API parameter" );
  CUGRAPH_EXPECTS( off != nullptr, "Invalid API parameter" );
  CUGRAPH_EXPECTS( ind != nullptr, "Invalid API parameter" );
  CUGRAPH_EXPECTS( val != nullptr, "Invalid API parameter" );
  CUGRAPH_EXPECTS( x_cols != nullptr, "Invalid API parameter" );
  CUGRAPH_EXPECTS( off->size > 0, "Invalid API parameter" );
  CUGRAPH_EXPECTS( ind->size > 0, "Invalid API parameter" );
  CUGRAPH_EXPECTS( val->size > 0, "Invalid API parameter" );
  CUGRAPH_EXPECTS( ind->size == val->size, "Column size mismatch" ); 
  CUGRAPH_EXPECTS( off->dtype == ind->dtype, "Unsupported data type" );  
  CUGRAPH_EXPECTS( off->null_count + ind->null_count + val->null_count == 0 , "Input column has non-zero null count");                 

  auto p = omp_get_num_threads();

  val_t* x[p];
  for (auto i = 0; i < p; ++i)
  {
    CUGRAPH_EXPECTS( x_cols[i] != nullptr, "Invalid API parameter" );
    CUGRAPH_EXPECTS( x_cols[i]->size > 0, "Invalid API parameter" );
    x[i]= static_cast<val_t*>(x_cols[i]->data);
  }
  #pragma omp master 
  { 
    cugraph::detail::Cusparse::get_handle();
  }
  SNMGinfo snmg_env;

  SNMGcsrmv<idx_t,val_t> spmv_solver(snmg_env, part_offsets,
                                      static_cast<idx_t*>(off->data), 
                                      static_cast<idx_t*>(ind->data), 
                                      static_cast<val_t*>(val->data), 
                                      x);
  spmv_solver.run(x);
  #pragma omp master 
  { 
    cugraph::detail::Cusparse::destroy_handle();
  }
  
}

} //namespace snmg

void snmg_csrmv (size_t * part_offsets, gdf_column * off, gdf_column * ind, gdf_column * val, gdf_column ** x_cols){
    switch (val->dtype) {
      case GDF_FLOAT32:   return snmg::snmg_csrmv_impl<int32_t,float>(part_offsets, off, ind, val, x_cols);
      case GDF_FLOAT64:   return snmg::snmg_csrmv_impl<int32_t,double>(part_offsets, off, ind, val, x_cols);
      default: CUGRAPH_FAIL("Unsupported data type");
    }
}
} //namespace cugraph