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
 * @file nvgraph_gdf.cu
 * ---------------------------------------------------------------------------**/

#include <nvgraph_gdf.h>
#include <ctime>
#include "utilities/error_utils.h"
#include "converters/nvgraph.cuh"

namespace cugraph {

  template <typename VT, typename WT>
  void createGraph_nvgraph(nvgraphHandle_t nvg_handle,
                                  Graph<VT,WT> *cugraph_G,
                                  nvgraphGraphDescr_t* nvg_G,
                                  bool use_transposed) {

  // check input
  CUGRAPH_EXPECTS(!((cugraph_G->edgeList == nullptr) &&
                  (cugraph_G->adjList == nullptr) &&
                  (cugraph_G->transposedAdjList == nullptr)),
              "Invalid API parameter");
  nvgraphTopologyType_t TT;
  cudaDataType_t settype;
  // create an nvgraph graph handle
  NVG_TRY(nvgraphCreateGraphDescr(nvg_handle, nvg_G));
  // setup nvgraph variables
  if (use_transposed) {
    // convert edgeList to transposedAdjList
    CUGRAPH_EXPECTS(cugraph_G->transposedAdjList != nullptr,
              "Invalid API parameter");
    // using exiting transposedAdjList if it exisits and if adjList is missing
    TT = NVGRAPH_CSC_32;
    nvgraphCSCTopology32I_st topoData;
    topoData.nvertices = cugraph_G->v;
    topoData.nedges = cugraph_G->e;
    topoData.destination_offsets = (int *) cugraph_G->transposedAdjList->offsets;
    topoData.source_indices = (int *) cugraph_G->transposedAdjList->indices;
    // attach the transposed adj list
    NVG_TRY(nvgraphAttachGraphStructure(nvg_handle, *nvg_G, (void * )&topoData, TT));
    //attach edge values
    if (cugraph_G->transposedAdjList->edge_data) {
      switch (typeid(cugraph_G->transposedAdjList->edge_data)) {
        case GDF_FLOAT32:
          settype = CUDA_R_32F;
          NVG_TRY(nvgraphAttachEdgeData(nvg_handle,
                                        *nvg_G,
                                        0,
                                        settype,
                                        (float * ) cugraph_G->transposedAdjList->edge_data))
          break;
        case GDF_FLOAT64:
          settype = CUDA_R_64F;
          NVG_TRY(nvgraphAttachEdgeData(nvg_handle,
                                        *nvg_G,
                                        0,
                                        settype,
                                        (double * ) cugraph_G->transposedAdjList->edge_data))
          break;
        default:
          CUGRAPH_FAIL("Unsupported data type");
      }
    }

  }
  else {
    CUGRAPH_EXPECTS(cugraph_G->adjList != nullptr,
              "Invalid API parameter");
    TT = NVGRAPH_CSR_32;
    nvgraphCSRTopology32I_st topoData;
    topoData.nvertices = cugraph_G->v;
    topoData.nedges = cugraph_G->e;
    topoData.source_offsets = (int *) cugraph_G->adjList->offsets;
     topoData.destination_indices = (int *) cugraph_G->adjList->indices;
 
    // attach adj list
    NVG_TRY(nvgraphAttachGraphStructure(nvg_handle, *nvg_G, (void * )&topoData, TT));
    //attach edge values
    if (cugraph_G->adjList->edge_data) {
      switch (typeid(cugraph_G->adjList->edge_data)) {
        case GDF_FLOAT32:
          settype = CUDA_R_32F;
          NVG_TRY(nvgraphAttachEdgeData(nvg_handle,
                                        *nvg_G,
                                        0,
                                        settype,
                                        (float * ) cugraph_G->adjList->edge_data))
          break;
        case GDF_FLOAT64:
          settype = CUDA_R_64F;
          NVG_TRY(nvgraphAttachEdgeData(nvg_handle,
                                        *nvg_G,
                                        0,
                                        settype,
                                        (double * ) cugraph_G->adjList->edge_data))
          break;
        default:
          CUGRAPH_FAIL("Unsupported data type");
      }
    }
  }
  
}

template void createGraph_nvgraph<int, float>(Graph<int, float> *cugraph_G, nvgraphGraphDescr_t* nvg_G, bool use_transposed);
template void createGraph_nvgraph<int, double>(Graph<int, double> *cugraph_G, nvgraphGraphDescr_t* nvg_G, bool use_transposed);

} // namespace
