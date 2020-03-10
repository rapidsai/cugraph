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
#include <algorithms.hpp>

using namespace hornets_nest;

namespace cugraph {

namespace detail {

template <typename VT, typename ET, typename WT>
void ktruss_subgraph_impl(experimental::GraphCOO<VT, ET, WT> const &graph,
                      int k,
                      experimental::GraphCOO<VT, ET, WT> &output_graph) {
  using HornetGraph = hornet::gpu::Hornet<VT>;
  using UpdatePtr   = hornet::BatchUpdatePtr<VT, hornet::EMPTY, hornet::DeviceType::DEVICE>;
  using Update      = hornet::gpu::BatchUpdate<VT>; cudaStream_t stream{nullptr};
  UpdatePtr ptr(graph.number_of_edges, graph.src_indices, graph.dst_indices);
  Update batch(ptr);

  HornetGraph hnt(graph.number_of_vertices+1);
  hnt.insert(batch);
  KTruss kt(hnt);

  kt.init();
  kt.reset();
  kt.createOffSetArray();
  kt.setInitParameters(4, 8, 2, 64000, 32);
  kt.reset();
  kt.sortHornet();

  kt.runForK(k);

  ET edge_count = kt.getGraphEdgeCount();
  ALLOC_TRY((void**)&output_graph.src_indices, sizeof(int) * edge_count, stream);
  ALLOC_TRY((void**)&output_graph.dst_indices, sizeof(int) * edge_count, stream);
  kt.copyGraph(output_graph.src_indices, output_graph.dst_indices);
  output_graph.number_of_vertices = graph.number_of_vertices;
  output_graph.number_of_edges = edge_count;
  output_graph.prop.directed = true;

  kt.release();
}

} // detail namespace

template <typename VT, typename ET, typename WT>
void k_truss_subgraph(experimental::GraphCOO<VT, ET, WT> const &graph,
                      int k,
                      experimental::GraphCOO<VT, ET, WT> &output_graph) {
  CUGRAPH_EXPECTS(graph->src_indices != nullptr, "Graph source indices cannot be a nullptr");
  CUGRAPH_EXPECTS(graph->dst_indices != nullptr, "Graph destination indices cannot be a nullptr");

  detail::ktruss_subgraph_impl(graph, k, output_graph);
}


}//namespace cugraph
