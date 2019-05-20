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
 
#include "spmv.cuh"

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
  cugraph::SNMGinfo snmg_env;
  cugraph::SNMGcsrmv<idx_t,val_t> spmv_solver(snmg_env, part_offsets,
                                      static_cast<idx_t*>(off->data), 
                                      static_cast<idx_t*>(ind->data), 
                                      static_cast<val_t*>(val->data), 
                                      x);
  spmv_solver.run(x);
  return GDF_SUCCESS;
}

gdf_error gdf_snmg_csrmv (size_t * part_offsets, gdf_column * off, gdf_column * ind, gdf_column * val, gdf_column ** x_cols){
    switch (val->dtype) {
      case GDF_FLOAT32:   return gdf_snmg_csrmv_impl<int32_t,float>(part_offsets, off, ind, val, x_cols);
      case GDF_FLOAT64:   return gdf_snmg_csrmv_impl<int32_t,double>(part_offsets, off, ind, val, x_cols);
      default: return GDF_UNSUPPORTED_DTYPE;
    }
}
