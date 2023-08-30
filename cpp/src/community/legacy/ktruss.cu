/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <cugraph/utilities/error.hpp>

#include "Static/KTruss/KTruss.cuh"
#include <Hornet.hpp>
#include <StandardAPI.hpp>
#include <cugraph/algorithms.hpp>

using namespace hornets_nest;

namespace cugraph {

namespace detail {

template <typename VT, typename ET, typename WT>
std::unique_ptr<legacy::GraphCOO<VT, ET, WT>> ktruss_subgraph_impl(
  legacy::GraphCOOView<VT, ET, WT> const& graph, int k, rmm::mr::device_memory_resource* mr)
{
  using HornetGraph = hornet::gpu::Hornet<VT>;
  using UpdatePtr   = hornet::BatchUpdatePtr<VT, hornet::EMPTY, hornet::DeviceType::DEVICE>;
  using Update      = hornet::gpu::BatchUpdate<VT>;
  cudaStream_t stream{nullptr};
  UpdatePtr ptr(graph.number_of_edges, graph.src_indices, graph.dst_indices);
  Update batch(ptr);

  HornetGraph hnt(graph.number_of_vertices + 1);
  hnt.insert(batch);
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to initialize graph");

  KTruss kt(hnt);

  kt.init();
  kt.reset();
  kt.createOffSetArray();
  // NOTE : These parameters will become obsolete once we move to the updated
  // algorithm (https://ieeexplore.ieee.org/document/8547581)
  kt.setInitParameters(4,      // Number of threads per block per list intersection
                       8,      // Number of intersections per block
                       2,      // log2(Number of threads)
                       64000,  // Total number of blocks launched
                       32);    // Thread block dimension
  kt.reset();
  kt.sortHornet();

  kt.runForK(k);
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to run");

  auto out_graph = std::make_unique<legacy::GraphCOO<VT, ET, WT>>(
    graph.number_of_vertices, kt.getGraphEdgeCount(), graph.has_data(), stream, mr);

  kt.copyGraph(out_graph->src_indices(), out_graph->dst_indices());

  kt.release();
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to release");

  return out_graph;
}
template <typename VT, typename ET, typename WT>
std::unique_ptr<legacy::GraphCOO<VT, ET, WT>> weighted_ktruss_subgraph_impl(
  legacy::GraphCOOView<VT, ET, WT> const& graph, int k, rmm::mr::device_memory_resource* mr)
{
  using HornetGraph = hornet::gpu::Hornet<VT, hornet::EMPTY, hornet::TypeList<WT>>;
  using UpdatePtr   = hornet::BatchUpdatePtr<VT, hornet::TypeList<WT>, hornet::DeviceType::DEVICE>;
  using Update      = hornet::gpu::BatchUpdate<VT, hornet::TypeList<WT>>;
  cudaStream_t stream{nullptr};
  UpdatePtr ptr(graph.number_of_edges, graph.src_indices, graph.dst_indices, graph.edge_data);
  Update batch(ptr);

  HornetGraph hnt(graph.number_of_vertices + 1);
  hnt.insert(batch);
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to initialize graph");

  KTrussWeighted<WT> kt(hnt);

  kt.init();
  kt.reset();
  kt.createOffSetArray();
  // NOTE : These parameters will become obsolete once we move to the updated
  // algorithm (https://ieeexplore.ieee.org/document/8547581)
  kt.setInitParameters(4,      // Number of threads per block per list intersection
                       8,      // Number of intersections per block
                       2,      // log2(Number of threads)
                       64000,  // Total number of blocks launched
                       32);    // Thread block dimension
  kt.reset();
  kt.sortHornet();

  kt.runForK(k);
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to run");

  auto out_graph = std::make_unique<legacy::GraphCOO<VT, ET, WT>>(
    graph.number_of_vertices, kt.getGraphEdgeCount(), graph.has_data(), stream, mr);

  kt.copyGraph(out_graph->src_indices(), out_graph->dst_indices(), out_graph->edge_data());

  kt.release();
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to release");

  return out_graph;
}

}  // namespace detail

template <typename VT, typename ET, typename WT>
std::unique_ptr<legacy::GraphCOO<VT, ET, WT>> k_truss_subgraph(
  legacy::GraphCOOView<VT, ET, WT> const& graph, int k, rmm::mr::device_memory_resource* mr)
{
  CUGRAPH_EXPECTS(graph.src_indices != nullptr, "Graph source indices cannot be a nullptr");
  CUGRAPH_EXPECTS(graph.dst_indices != nullptr, "Graph destination indices cannot be a nullptr");

  if (graph.edge_data == nullptr) {
    return detail::ktruss_subgraph_impl(graph, k, mr);
  } else {
    return detail::weighted_ktruss_subgraph_impl(graph, k, mr);
  }
}

template std::unique_ptr<legacy::GraphCOO<int32_t, int32_t, float>>
k_truss_subgraph<int, int, float>(legacy::GraphCOOView<int, int, float> const&,
                                  int,
                                  rmm::mr::device_memory_resource*);

template std::unique_ptr<legacy::GraphCOO<int32_t, int32_t, double>>
k_truss_subgraph<int, int, double>(legacy::GraphCOOView<int, int, double> const&,
                                   int,
                                   rmm::mr::device_memory_resource*);

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
k_truss_subgraph(raft::handle_t const& handle,
                 raft::device_span<vertex_t const> src,
                 raft::device_span<vertex_t const> dst,
                 std::optional<raft::device_span<weight_t const>> wgt,
                 size_t number_of_vertices,
                 int k)
{
  using HornetGraph = hornet::gpu::Hornet<vertex_t>;
  using UpdatePtr   = hornet::BatchUpdatePtr<vertex_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
  using Update      = hornet::gpu::BatchUpdate<vertex_t>;

  HornetGraph hnt(number_of_vertices + 1);

  if (wgt) {
    UpdatePtr ptr(src.size(), src.data(), dst.data(), wgt->data());
    Update batch(ptr);

    hnt.insert(batch);
    CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to initialize graph");
  } else {
    UpdatePtr ptr(src.size(), src.data(), dst.data());
    Update batch(ptr);

    hnt.insert(batch);
    CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to initialize graph");
  }

  KTruss kt(hnt);

  kt.init();
  kt.reset();
  kt.createOffSetArray();
  // NOTE : These parameters will become obsolete once we move to the updated
  // algorithm (https://ieeexplore.ieee.org/document/8547581)
  kt.setInitParameters(4,      // Number of threads per block per list intersection
                       8,      // Number of intersections per block
                       2,      // log2(Number of threads)
                       64000,  // Total number of blocks launched
                       32);    // Thread block dimension
  kt.reset();
  kt.sortHornet();

  kt.runForK(k);
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to run");

  rmm::device_uvector<vertex_t> result_src(kt.getGraphEdgeCount(), handle.get_stream());
  rmm::device_uvector<vertex_t> result_dst(kt.getGraphEdgeCount(), handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> result_wgt{std::nullopt};

  if (wgt) {
    result_wgt = rmm::device_uvector<weight_t>(kt.getGraphEdgeCount(), handle.get_stream());
    kt.copyGraph(result_src.data(), result_dst.data(), result_wgt->data());
  } else {
    kt.copyGraph(result_src.data(), result_dst.data());
  }

  kt.release();
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to release");

  return out_graph;
}

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
k_truss_subgraph(raft::handle_t const& handle,
                 raft::device_span<int32_t const> src,
                 raft::device_span<int32_t const> dst,
                 std::optional<raft::device_span<float const>> wgt,
                 size_t number_of_vertices,
                 int k);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
k_truss_subgraph(raft::handle_t const& handle,
                 raft::device_span<int32_t const> src,
                 raft::device_span<int32_t const> dst,
                 std::optional<raft::device_span<double const>> wgt,
                 size_t number_of_vertices,
                 int k);

}  // namespace cugraph
