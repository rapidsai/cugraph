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
 * @brief Core Number implementation
 *
 * @file katz_centrality.cu
 * --------------------------------------------------------------------------*/

#include <cugraph.h>
#include "utilities/error_utils.h"
#include <Hornet.hpp>
#include <Static/KatzCentrality/Katz.cuh>

gdf_error gdf_katz_centrality(gdf_graph *graph,
                              gdf_column *katz_centrality,
                              double alpha,
                              int max_iter,
                              double tol,
                              bool has_guess,
                              bool normalized) {
  GDF_REQUIRE(graph->adjList != nullptr || graph->edgeList != nullptr, GDF_INVALID_API_CALL);
  gdf_error err = gdf_add_adj_list(graph);
  if (err != GDF_SUCCESS)
    return err;
  GDF_REQUIRE(graph->adjList->offsets->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
  GDF_REQUIRE(graph->adjList->indices->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);

  const bool isStatic = true;
  using HornetGraph = hornet::gpu::HornetStatic<int>;
  using HornetInit  = hornet::HornetInit<int>;
  using Katz = hornets_nest::KatzCentralityStatic;
  HornetInit init(graph->numberOfVertices, graph->adjList->indices->size,
      reinterpret_cast<int*>(graph->adjList->offsets->data),
      reinterpret_cast<int*>(graph->adjList->indices->data));
  HornetGraph hnt(init, hornet::DeviceType::DEVICE);
  Katz katz(hnt, alpha, max_iter, tol, normalized, isStatic, reinterpret_cast<double*>(katz_centrality->data));
  if (katz.getAlpha() < alpha) {
    std::cerr<<"Error : alpha is not small enough ( < "<<katz.getAlpha()<<") for convergence"<<std::endl; return GDF_CUDA_ERROR;
  }
  katz.run();
  if (!katz.hasConverged()) {
    std::cerr<<"Error : Convergence not reached"<<std::endl; return GDF_CUDA_ERROR;
  }
  return GDF_SUCCESS;
}
