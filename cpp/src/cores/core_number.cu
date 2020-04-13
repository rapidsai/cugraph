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

#include <graph.hpp>
#include "utilities/error_utils.h"
#include <Hornet.hpp>
#include <Static/CoreNumber/CoreNumber.cuh>
#include <rmm_utils.h>
//#include <nvgraph_gdf.h>

namespace cugraph {
namespace detail {

template <typename VT, typename ET, typename WT>
void core_number(experimental::GraphCSR<VT, ET, WT> const &graph,
                 int *core_number) {

  using HornetGraph = hornet::gpu::HornetStatic<int>;
  using HornetInit  = hornet::HornetInit<VT>;
  using CoreNumber  = hornets_nest::CoreNumberStatic;
  HornetInit init(graph.number_of_vertices,
                  graph.number_of_edges,
                  graph.offsets,
                  graph.indices);
  HornetGraph hnt(init, hornet::DeviceType::DEVICE);
  CoreNumber cn(hnt, core_number);
  cn.run();
}

struct FilterEdges {
  int k;
  int* core_number;

  FilterEdges(int _k, int *d_core_num) :
    k(_k), core_number(d_core_num) {}

  template <typename T>
  __host__ __device__
    bool operator()(T t) {
      int src = thrust::get<0>(t);
      int dst = thrust::get<1>(t);
      return (core_number[src] >= k) && (core_number[dst] >= k);
    }
};

template <typename VT, typename ET, typename WT>
void extract_edges(experimental::GraphCOO<VT, ET, WT> const &i_graph,
                   experimental::GraphCOO<VT, ET, WT> &o_graph,
                   VT *d_core,
                   int k,
                   ET filteredEdgeCount) {

  cudaStream_t stream{nullptr};

  ALLOC_TRY(&o_graph.src_indices, sizeof(VT) * filteredEdgeCount, stream);
  ALLOC_TRY(&o_graph.dst_indices, sizeof(VT) * filteredEdgeCount, stream);
  o_graph.edge_data = nullptr;

  bool hasData = (i_graph.edge_data != nullptr);


  //If an edge satisfies k-core conditions i.e. core_num[src] and core_num[dst]
  //are both greater than or equal to k, copy it to the output graph
  if (hasData) {
    ALLOC_TRY(&o_graph.edge_data, sizeof(WT) * filteredEdgeCount, stream);

    auto inEdge = thrust::make_zip_iterator(thrust::make_tuple(i_graph.src_indices,
                                                               i_graph.dst_indices,
                                                               i_graph.edge_data));
    auto outEdge = thrust::make_zip_iterator(thrust::make_tuple(o_graph.src_indices,
                                                                o_graph.dst_indices,
                                                                o_graph.edge_data));
    auto ptr = thrust::copy_if(rmm::exec_policy(stream)->on(stream),
                               inEdge, inEdge + i_graph.number_of_edges,
                               outEdge,
                               FilterEdges(k, d_core));
    if (thrust::distance(outEdge, ptr) != filteredEdgeCount) { CUGRAPH_FAIL("Edge extraction failed"); }
  } else {
    auto inEdge = thrust::make_zip_iterator(thrust::make_tuple(i_graph.src_indices,
                                                               i_graph.dst_indices));
    auto outEdge = thrust::make_zip_iterator(thrust::make_tuple(o_graph.src_indices,
                                                                o_graph.dst_indices));
    auto ptr = thrust::copy_if(rmm::exec_policy(stream)->on(stream),
                               inEdge, inEdge + i_graph.number_of_edges,
                               outEdge,
                               FilterEdges(k, d_core));
    if (thrust::distance(outEdge, ptr) != filteredEdgeCount) { CUGRAPH_FAIL("Edge extraction failed"); }
  }
}

//Extract a subgraph from in_graph (with or without weights)
//to out_graph based on whether edges in in_graph satisfy kcore
//conditions.
//i.e. All edges (s,d,w) in in_graph are copied over to out_graph
//if core_num[s] and core_num[d] are greater than or equal to k.
template <typename VT, typename ET, typename WT>
void extract_subgraph(experimental::GraphCOO<VT, ET, WT> const &in_graph,
                      experimental::GraphCOO<VT, ET, WT> &out_graph,
                      int const *vid,
                      int const *core_num,
                      int k,
                      int len,
                      int num_verts) {

  cudaStream_t stream{nullptr};

  rmm::device_vector<VT> sorted_core_num(num_verts);

  thrust::scatter(rmm::exec_policy(stream)->on(stream),
                  core_num, core_num + len,
                  vid, sorted_core_num.begin());

  VT *d_sorted_core_num = sorted_core_num.data().get();

  //Count number of edges in the input graph that satisfy kcore conditions
  //i.e. core_num[src] and core_num[dst] are both greater than or equal to k
  auto edge = thrust::make_zip_iterator(thrust::make_tuple(in_graph.src_indices,
                                                           in_graph.dst_indices));

  out_graph.number_of_vertices = in_graph.number_of_vertices;

  out_graph.number_of_edges = thrust::count_if(rmm::exec_policy(stream)->on(stream),
                                               edge, edge + in_graph.number_of_edges,
                                               detail::FilterEdges(k, d_sorted_core_num));

  return extract_edges<VT,ET,WT>(in_graph, out_graph, d_sorted_core_num, k, out_graph.number_of_edges);
}

} //namespace detail


template <typename VT, typename ET, typename WT>
void core_number(experimental::GraphCSR<VT, ET, WT> const &graph, VT *core_number) {
  return detail::core_number(graph, core_number);
}

template <typename VT, typename ET, typename WT>
void k_core(experimental::GraphCOO<VT, ET, WT> const &in_graph,
            int k,
            VT const *vertex_id,
            VT const *core_number,
            VT num_vertex_ids,
            experimental::GraphCOO<VT, ET, WT> &out_graph) {

  CUGRAPH_EXPECTS(vertex_id != nullptr, "Invalid API parameter: vertex_id is NULL");
  CUGRAPH_EXPECTS(core_number != nullptr, "Invalid API parameter: core_number is NULL");
  CUGRAPH_EXPECTS(k >= 0, "Invalid API parameter: k must be >= 0");

  detail::extract_subgraph(in_graph, out_graph,
                           vertex_id, core_number,
                           k, num_vertex_ids, in_graph.number_of_vertices);
}

template void core_number<int32_t, int32_t, float>(experimental::GraphCSR<int32_t, int32_t, float> const &, int32_t *core_number);
template void k_core<int32_t, int32_t, float>(experimental::GraphCOO<int32_t, int32_t, float> const &, int, int32_t const *,
                                              int32_t const *, int32_t, experimental::GraphCOO<int32_t, int32_t, float> &);
template void k_core<int32_t, int32_t, double>(experimental::GraphCOO<int32_t, int32_t, double> const &, int, int32_t const *,
                                               int32_t const *, int32_t, experimental::GraphCOO<int32_t, int32_t, double> &);

} //namespace cugraph
