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

/**
 * ---------------------------------------------------------------------------*
 * @brief KTruss implementation
 *
 * @file ktruss.cu
 * --------------------------------------------------------------------------*/


#include <cugraph.h>
#include "utilities/error_utils.h"
#include <Hornet.hpp>
#include "Static/KTruss/KTruss.cuh"
#include <StandardAPI.hpp>
#include <rmm_utils.h>
#include <nvgraph_gdf.h>

using namespace hornets_nest;


namespace cugraph {
namespace detail {

void ktruss_max_impl(gdf_graph *graph,
                          int *k_max) {
  // gdf_error err = gdf_add_adj_list(graph);
  // if (err != GDF_SUCCESS)
  //   return err;
  int * src = static_cast<int*>(graph->edgeList->src_indices->data);
  int * dst = static_cast<int*>(graph->edgeList->dest_indices->data);
  using HornetGraph = hornet::gpu::Hornet<int>;
  using UpdatePtr   = ::hornet::BatchUpdatePtr<int, hornet::EMPTY, hornet::DeviceType::DEVICE>;
  using Update      = ::hornet::gpu::BatchUpdate<int>;

  UpdatePtr ptr(graph->edgeList->src_indices->size, src, dst);
  Update batch(ptr);
  gdf_number_of_vertices(graph);

  HornetGraph hnt(graph->numberOfVertices);
  hnt.insert(batch);
  //Use hornet

 
  KTruss kt(hnt);

  kt.init();
  kt.reset();

  kt.createOffSetArray();
  kt.setInitParameters(4, 8, 2, 64000, 32);
  kt.reset(); 
  kt.sortHornet();

  kt.run();

  *k_max = kt.getMaxK();

  kt.release();
}

void k_truss_max(gdf_graph *graph,
                          int *k_max) {
  CUGRAPH_EXPECTS(graph->edgeList->src_indices != nullptr || graph->edgeList->dest_indices != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(graph->edgeList->src_indices->data != nullptr || graph->edgeList->dest_indices->data != nullptr, "Invalid API parameter");

  CUGRAPH_EXPECTS(graph->edgeList->src_indices->dtype == GDF_INT32, "Unsupported data type");
  CUGRAPH_EXPECTS(graph->edgeList->dest_indices->dtype == GDF_INT32, "Unsupported data type");

  ktruss_max_impl(graph, k_max);
}

void ktruss_subgraph_impl(gdf_graph *graph,
                          int k,
                          gdf_graph *ktrussgraph) {
  int * src = static_cast<int*>(graph->edgeList->src_indices->data);
  int * dst = static_cast<int*>(graph->edgeList->dest_indices->data);
  using HornetGraph = hornet::gpu::Hornet<int>;
  using UpdatePtr   = ::hornet::BatchUpdatePtr<int, hornet::EMPTY, hornet::DeviceType::DEVICE>;
  using Update      = ::hornet::gpu::BatchUpdate<int>;


  UpdatePtr ptr(graph->edgeList->src_indices->size, src, dst);
  Update batch(ptr);
  gdf_number_of_vertices(graph);
  HornetGraph hnt(graph->numberOfVertices);
  hnt.insert(batch);

  
  // KTruss kt(hnt);

  // kt.init();
  // kt.reset();

  // kt.createOffSetArray();
  // kt.setInitParameters(4, 8, 2, 64000, 32);
  // kt.reset(); 
  // kt.sortHornet();

  // printf("Number of edges before %d : ",hnt.nE());

  // // kt.runForK(k);

  // printf("Number of edges after  %d : ",hnt.nE());


  // kt.release();
  
}

void k_truss_subgraph(gdf_graph *graph,
                          int k,
                          gdf_graph *ktrussgraph) {

  CUGRAPH_EXPECTS(graph->edgeList->src_indices != nullptr || graph->edgeList->dest_indices != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(graph->edgeList->src_indices->data != nullptr || graph->edgeList->dest_indices->data != nullptr, "Invalid API parameter");

  CUGRAPH_EXPECTS(graph->edgeList->src_indices->dtype == GDF_INT32, "Unsupported data type");
  CUGRAPH_EXPECTS(graph->edgeList->dest_indices->dtype == GDF_INT32, "Unsupported data type");

  ktruss_subgraph_impl(graph, k,NULL);
}


}
}//namespace cugraph