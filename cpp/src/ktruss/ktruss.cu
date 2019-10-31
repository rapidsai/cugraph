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
#include <Static/KTruss/KTruss.cuh>
#include <rmm_utils.h>
#include <nvgraph_gdf.h>

gdf_error ktruss_max_impl(gdf_graph *graph,
                          int *core_number) {
  gdf_error err = gdf_add_adj_list(graph);
  if (err != GDF_SUCCESS)
    return err;
  using HornetGraph = hornet::gpu::Hornet<int>;
  using HornetInit  = hornet::HornetInit<int>;
  // using KTruss  = hornets_nest::KTruss<HornetGraph>;
  HornetInit init(graph->numberOfVertices, graph->adjList->indices->size,
      static_cast<int*>(graph->adjList->offsets->data),
      static_cast<int*>(graph->adjList->indices->data));
  HornetGraph hnt(init, hornet::DeviceType::DEVICE);
  // KTruss kt(hnt);
  // kt.run();
  return GDF_SUCCESS;
}

gdf_error gdf_k_truss_max(gdf_graph *graph,
                          int *k_max) {
  GDF_REQUIRE(graph->adjList != nullptr || graph->edgeList != nullptr, GDF_INVALID_API_CALL);
  gdf_error err = gdf_add_adj_list(graph);
  if (err != GDF_SUCCESS)
    return err;
  GDF_REQUIRE(graph->adjList->offsets->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
  GDF_REQUIRE(graph->adjList->indices->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);

  return ktruss_max_impl(graph, k_max);
}

