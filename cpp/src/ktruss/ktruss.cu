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
  using Update      = hornet::gpu::BatchUpdate<VT>;
  VT * src = const_cast<VT*>(graph.src_indices);
  VT * dst = const_cast<VT*>(graph.dst_indices);
  cudaStream_t stream{nullptr};
  UpdatePtr ptr(graph.number_of_edges, src, dst);
  Update batch(ptr);

  HornetGraph hnt(graph.number_of_vertices+1);
  hnt.insert(batch);
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to initialize graph");

  KTruss kt(hnt);

  kt.init();
  kt.reset();
  kt.createOffSetArray();
  //NOTE : These parameters will become obsolete once we move to the updated
  //algorithm (https://ieeexplore.ieee.org/document/8547581)
  kt.setInitParameters(
      4,//Number of threads per block per list intersection
      8,//Number of intersections per block
      2,//log2(Number of threads)
      64000,//Total number of blocks launched
      32//Thread block dimension);
  kt.reset();
  kt.sortHornet();

  kt.runForK(k);
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to run");

  ET subgraph_edge_count = kt.getGraphEdgeCount();

  VT * out_src;
  VT * out_dst;
  ALLOC_TRY((void**)&out_src, sizeof(VT) * subgraph_edge_count, stream);
  ALLOC_TRY((void**)&out_dst, sizeof(VT) * subgraph_edge_count, stream);

  kt.copyGraph(out_src, out_dst);

  experimental::GraphCOO<VT, ET, WT> subgraph(
      const_cast<const VT*>(out_src), const_cast<const VT*>(out_dst),
      nullptr, graph.number_of_vertices, subgraph_edge_count);

  output_graph = subgraph;
  output_graph.prop.directed = true;
  kt.release();
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to release");
}

} // detail namespace

template <typename VT, typename ET, typename WT>
void k_truss_subgraph(experimental::GraphCOO<VT, ET, WT> const &graph,
                      int k,
                      experimental::GraphCOO<VT, ET, WT> &output_graph) {
  CUGRAPH_EXPECTS(graph.src_indices != nullptr, "Graph source indices cannot be a nullptr");
  CUGRAPH_EXPECTS(graph.dst_indices != nullptr, "Graph destination indices cannot be a nullptr");

  detail::ktruss_subgraph_impl(graph, k, output_graph);
}

template void k_truss_subgraph<int, int, float>(experimental::GraphCOO<int, int, float> const &graph,
    int k, experimental::GraphCOO<int, int, float> &output_graph);
template void k_truss_subgraph<int, int, double>(experimental::GraphCOO<int, int, double> const &graph,
    int k, experimental::GraphCOO<int, int, double> &output_graph);

}//namespace cugraph
