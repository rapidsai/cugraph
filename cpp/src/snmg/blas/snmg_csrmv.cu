#include <cugraph.h>
#include "utilities/error_utils.h"
#include "snmg/blas/spmv.cuh"

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

  gdf_error status;
  auto p = omp_get_num_threads();

  val_t* x[p];
  for (auto i = 0; i < p; ++i)
  {
    GDF_REQUIRE( x_cols[i] != nullptr, GDF_INVALID_API_CALL );
    GDF_REQUIRE( x_cols[i]->size > 0, GDF_INVALID_API_CALL );
    x[i]= static_cast<val_t*>(x_cols[i]->data);
  }
  status = cugraph::snmg_csrmv<idx_t,val_t>(part_offsets,
                                      static_cast<idx_t*>(off->data),
                                      static_cast<idx_t*>(ind->data),
                                      static_cast<val_t*>(val->data),
                                      x);
  return status;
}

gdf_error gdf_snmg_csrmv (size_t * part_offsets, gdf_column * off, gdf_column * ind, gdf_column * val, gdf_column ** x_cols){
    switch (val->dtype) {
      case GDF_FLOAT32:   return gdf_snmg_csrmv_impl<int32_t,float>(part_offsets, off, ind, val, x_cols);
      case GDF_FLOAT64:   return gdf_snmg_csrmv_impl<int32_t,double>(part_offsets, off, ind, val, x_cols);
      default: return GDF_UNSUPPORTED_DTYPE;
    }
}
