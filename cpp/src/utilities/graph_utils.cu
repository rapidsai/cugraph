/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// Interanl helper functions 
#include "utilities/graph_utils.cuh"

namespace cugraph { 
namespace detail {
    
void gdf_col_set_defaults(gdf_column* col) {
  col->dtype = GDF_invalid;
  col->size = 0;
  col->data = nullptr;
  col->valid = nullptr;
  col->null_count = 0;
  gdf_dtype_extra_info extra_info;
  extra_info.time_unit = TIME_UNIT_NONE;
  col->dtype_info = extra_info;  
}

} } //namespace
