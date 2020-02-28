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

void ktruss_max_impl(Graph *graph,
                     int *k_max) {
  int * src = static_cast<int*>(graph->edgeList->src_indices->data);
  int * dst = static_cast<int*>(graph->edgeList->dest_indices->data);

  using HornetGraph = hornet::gpu::Hornet<int>;
  using UpdatePtr   = ::hornet::BatchUpdatePtr<int, hornet::EMPTY, hornet::DeviceType::DEVICE>;
  using Update      = ::hornet::gpu::BatchUpdate<int>;

  UpdatePtr ptr(graph->edgeList->src_indices->size, src, dst);
  Update batch(ptr);
  number_of_vertices(graph);

  HornetGraph hnt(graph->numberOfVertices+1);
  hnt.insert(batch);

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

} // detail namespace

void k_truss_max(Graph *graph,
                          int *k_max) {
  CUGRAPH_EXPECTS(graph->edgeList->src_indices != nullptr ||
      graph->edgeList->dest_indices != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(graph->edgeList->src_indices->data != nullptr ||
      graph->edgeList->dest_indices->data != nullptr, "Invalid API parameter");

  CUGRAPH_EXPECTS(graph->adjList->offsets != nullptr ||
      graph->edgeList->dest_indices != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(graph->adjList->offsets->data != nullptr ||
      graph->edgeList->dest_indices != nullptr, "Invalid API parameter");

  CUGRAPH_EXPECTS(graph->adjList->indices != nullptr ||
      graph->edgeList->dest_indices != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(graph->adjList->indices->data != nullptr ||
      graph->edgeList->dest_indices != nullptr, "Invalid API parameter");


  CUGRAPH_EXPECTS(graph->edgeList->src_indices->dtype == GDF_INT32,
      "Unsupported data type");
  CUGRAPH_EXPECTS(graph->edgeList->dest_indices->dtype == GDF_INT32,
      "Unsupported data type");
  CUGRAPH_EXPECTS(graph->adjList->offsets->dtype == GDF_INT32,
      "Unsupported data type");
  CUGRAPH_EXPECTS(graph->adjList->indices->dtype == GDF_INT32,
      "Unsupported data type");

  detail::ktruss_max_impl(graph, k_max);
}

namespace detail {

void createOutputGraph(Graph *ktrussgraph, KTruss& kt) {
  cudaStream_t stream{nullptr};

  //Allocate output columns
  ktrussgraph->edgeList = new gdf_edge_list;
  ktrussgraph->edgeList->src_indices = new gdf_column;
  ktrussgraph->edgeList->dest_indices = new gdf_column;
  ktrussgraph->edgeList->ownership = 2;

  int edge_count = kt.getGraphEdgeCount();

  int *o_src, *o_dst;
  ALLOC_TRY((void**)&o_src, sizeof(int) * edge_count, stream);
  ALLOC_TRY((void**)&o_dst, sizeof(int) * edge_count, stream);

  kt.copyGraph(o_src, o_dst);

  gdf_column_view(ktrussgraph->edgeList->src_indices, o_src,
      nullptr, edge_count, GDF_INT32);
  gdf_column_view(ktrussgraph->edgeList->dest_indices, o_dst,
      nullptr, edge_count, GDF_INT32);
}

void ktruss_subgraph_impl(Graph *graph,
                          int k,
                          Graph *ktrussgraph) {
  int * src = static_cast<int*>(graph->edgeList->src_indices->data);
  int * dst = static_cast<int*>(graph->edgeList->dest_indices->data);
  using HornetGraph = hornet::gpu::Hornet<int>;
  using UpdatePtr   = ::hornet::BatchUpdatePtr<int, hornet::EMPTY,
      hornet::DeviceType::DEVICE>;
  using Update      = ::hornet::gpu::BatchUpdate<int>;

  UpdatePtr ptr(graph->edgeList->src_indices->size, src, dst);
  Update batch(ptr);
  number_of_vertices(graph);
  HornetGraph hnt(graph->numberOfVertices);
  hnt.insert(batch);

  KTruss kt(hnt);

  kt.init();
  kt.reset();

  kt.createOffSetArray();
  kt.setInitParameters(4, 8, 2, 64000, 32);
  kt.reset();
  kt.sortHornet();

  kt.runForK(k);

  createOutputGraph(ktrussgraph, kt);

  kt.release();
}
} // detail namespace

void k_truss_subgraph(Graph *graph,
                          int k,
                          Graph *ktrussgraph) {

  CUGRAPH_EXPECTS(graph->edgeList->src_indices != nullptr ||
      graph->edgeList->dest_indices != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(graph->edgeList->src_indices->data != nullptr ||
      graph->edgeList->dest_indices->data != nullptr, "Invalid API parameter");

  CUGRAPH_EXPECTS(graph->edgeList->src_indices->dtype ==  GDF_INT32,
      "Unsupported data type");
  CUGRAPH_EXPECTS(graph->edgeList->dest_indices->dtype == GDF_INT32,
      "Unsupported data type");

  detail::ktruss_subgraph_impl(graph, k, ktrussgraph);
}


}//namespace cugraph
