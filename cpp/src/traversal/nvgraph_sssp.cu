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
/** ---------------------------------------------------------------------------*
 * @brief Wrapper functions for Nvgraph
 *
 * @file nvgraph_sssp.cu
 * ---------------------------------------------------------------------------**/

#include <nvgraph_gdf.h>
#include <nvgraph/nvgraph.h>
#include <thrust/device_vector.h>
#include "utilities/error_utils.h"
#include "converters/nvgraph.cuh"
#include <rmm_utils.h>

template<typename T>
using Vector = thrust::device_vector<T, rmm_allocator<T>>;

gdf_error gdf_sssp_nvgraph(gdf_graph *gdf_G,
                            const int *source_vert,
                            gdf_column *sssp_distances) {

  GDF_REQUIRE(gdf_G != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(*source_vert >= 0, GDF_INVALID_API_CALL);
  GDF_REQUIRE(*source_vert < sssp_distances->size, GDF_INVALID_API_CALL);
  GDF_REQUIRE(sssp_distances != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(sssp_distances->data != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(!sssp_distances->valid, GDF_VALIDITY_UNSUPPORTED);
  GDF_REQUIRE(sssp_distances->size > 0, GDF_INVALID_API_CALL);

  // init nvgraph
  // TODO : time this call
  nvgraphHandle_t nvg_handle = 0;
  nvgraphGraphDescr_t nvgraph_G = 0;
  cudaDataType_t settype;

  NVG_TRY(nvgraphCreate(&nvg_handle));
  GDF_TRY(gdf_createGraph_nvgraph(nvg_handle, gdf_G, &nvgraph_G, true));

  int sssp_index = 0;
  int weight_index = 0;
  Vector<float> d_val;

  //RMM:
  //        
  cudaStream_t stream { nullptr };
  rmm_temp_allocator allocator(stream);
  if (gdf_G->transposedAdjList->edge_data == nullptr) {
    // use a fp32 vector  [1,...,1]
    settype = CUDA_R_32F;
    d_val.resize(gdf_G->transposedAdjList->indices->size);
    thrust::fill(thrust::cuda::par(allocator).on(stream), d_val.begin(), d_val.end(), 1.0);
    NVG_TRY(nvgraphAttachEdgeData(nvg_handle,
                                  nvgraph_G,
                                  weight_index,
                                  settype,
                                  (void * ) thrust::raw_pointer_cast(d_val.data())));
  }
  else {
    switch (gdf_G->transposedAdjList->edge_data->dtype) {
      case GDF_FLOAT32:
        settype = CUDA_R_32F;
        break;
      case GDF_FLOAT64:
        settype = CUDA_R_64F;
        break;
      default:
        return GDF_UNSUPPORTED_DTYPE;
    }
  }

  NVG_TRY(nvgraphAttachVertexData(nvg_handle, nvgraph_G, 0, settype, sssp_distances->data));

  NVG_TRY(nvgraphSssp(nvg_handle, nvgraph_G, weight_index, source_vert, sssp_index));

  NVG_TRY(nvgraphDestroyGraphDescr(nvg_handle, nvgraph_G));
  NVG_TRY(nvgraphDestroy(nvg_handle));

  return GDF_SUCCESS;
}
