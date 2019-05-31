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


namespace cugraph
{
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
  cudaCheckError();
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
   
  allgather (env, part_off, y_loc, x);
}

template class SNMGcsrmv<int, double>;
template class SNMGcsrmv<int, float>;

template <typename idx_t,typename val_t>
gdf_error gdf_snmg_csrmv_impl (size_t * part_offsets, gdf_column * off, gdf_column * ind, gdf_column * val, gdf_column ** x_cols){
  
  GDF_REQUIRE( part_offsets != nullptr, GDF_INVALID_API_CALL );
  GDF_REQUIRE( off != nullptr, GDF_INVALID_API_CALL );
  GDF_REQUIRE( ind != nullptr, GDF_INVALID_API_CALL );
  GDF_REQUIRE( val != nullptr, GDF_INVALID_API_CALL );
  GDF_REQUIRE( x_cols != nullptr, GDF_INVALID_API_CALL );
  GDF_REQUIRE( off->size > 0, GDF_INVALID_API_CALL );
  GDF_REQUIRE( ind->size > 0, GDF_INVALID_API_CALL );
  GDF_REQUIRE( val->size > 0, GDF_INVALID_API_CALL );
  GDF_REQUIRE( ind->size == val->size, GDF_COLUMN_SIZE_MISMATCH ); 
  GDF_REQUIRE( off->dtype == ind->dtype, GDF_UNSUPPORTED_DTYPE );  
  GDF_REQUIRE( off->null_count + ind->null_count + val->null_count == 0 , GDF_VALIDITY_UNSUPPORTED );                 

  auto p = omp_get_num_threads();

  val_t* x[p];
  for (auto i = 0; i < p; ++i)
  {
    GDF_REQUIRE( x_cols[i] != nullptr, GDF_INVALID_API_CALL );
    GDF_REQUIRE( x_cols[i]->size > 0, GDF_INVALID_API_CALL );
    x[i]= static_cast<val_t*>(x_cols[i]->data);
  }
  #pragma omp master 
  { 
    Cusparse::get_handle();
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
    Cusparse::destroy_handle();
  }
  return GDF_SUCCESS;
}

} //namespace

gdf_error gdf_snmg_csrmv (size_t * part_offsets, gdf_column * off, gdf_column * ind, gdf_column * val, gdf_column ** x_cols){
    switch (val->dtype) {
      case GDF_FLOAT32:   return cugraph::gdf_snmg_csrmv_impl<int32_t,float>(part_offsets, off, ind, val, x_cols);
      case GDF_FLOAT64:   return cugraph::gdf_snmg_csrmv_impl<int32_t,double>(part_offsets, off, ind, val, x_cols);
      default: return GDF_UNSUPPORTED_DTYPE;
    }
}
