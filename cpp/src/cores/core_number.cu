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
 * @file core_number.cu
 * --------------------------------------------------------------------------*/

#include <cugraph.h>
#include "utilities/error_utils.h"
#include <Hornet.hpp>
#include <Static/CoreNumber/CoreNumber.cuh>
#include <rmm_utils.h>
#include <nvgraph_gdf.h>

namespace cugraph {
namespace detail {

void core_number_impl(Graph *graph,
                          int *core_number) {
  using HornetGraph = hornet::gpu::HornetStatic<int>;
  using HornetInit  = hornet::HornetInit<int>;
  using CoreNumber  = hornets_nest::CoreNumberStatic;
  HornetInit init(graph->numberOfVertices, graph->adjList->indices->size,
      static_cast<int*>(graph->adjList->offsets->data),
      static_cast<int*>(graph->adjList->indices->data));
  HornetGraph hnt(init, hornet::DeviceType::DEVICE);
  CoreNumber cn(hnt, core_number);
  cn.run();
  
}

struct FilterEdges {
  int k;
  int* core_number;

  FilterEdges(int _k, thrust::device_ptr<int> core_num) :
    k(_k), core_number(core_num.get()) {}

  template <typename T>
  __host__ __device__
    bool operator()(T t) {
      int src = thrust::get<0>(t);
      int dst = thrust::get<1>(t);
      return (core_number[src] >= k) && (core_number[dst] >= k);
    }
};

template <typename WT>
void extract_edges(
    Graph *i_graph,
    Graph *o_graph,
    thrust::device_ptr<int> c_ptr,
    int k,
    int filteredEdgeCount) {
  cudaStream_t stream{nullptr};

  //Allocate output columns
  o_graph->edgeList = new gdf_edge_list;
  o_graph->edgeList->src_indices = new gdf_column;
  o_graph->edgeList->dest_indices = new gdf_column;
  o_graph->edgeList->ownership = 2;

  bool hasData = (i_graph->edgeList->edge_data != nullptr);

  //Allocate underlying memory for output columns
  int *o_src, *o_dst, *o_wgt;
  ALLOC_TRY((void**)&o_src, sizeof(int) * filteredEdgeCount, stream);
  ALLOC_TRY((void**)&o_dst, sizeof(int) * filteredEdgeCount, stream);

  int *i_src = static_cast<int*>(i_graph->edgeList->src_indices->data);
  int *i_dst = static_cast<int*>(i_graph->edgeList->dest_indices->data);
  WT  *i_wgt = nullptr;

  gdf_column_view(o_graph->edgeList->src_indices, o_src,
      nullptr, filteredEdgeCount, GDF_INT32);
  gdf_column_view(o_graph->edgeList->dest_indices, o_dst,
      nullptr, filteredEdgeCount, GDF_INT32);

  //Set pointers and allocate memory/columns in case input graph has edge_data
  if (hasData) {
    o_graph->edgeList->edge_data   = new gdf_column;
    ALLOC_TRY((void**)&o_wgt, sizeof(WT)  * filteredEdgeCount, stream);
    i_wgt = static_cast<WT*>(i_graph->edgeList->edge_data->data);
    gdf_column_view(o_graph->edgeList->edge_data,   o_wgt,
        nullptr, filteredEdgeCount, i_graph->edgeList->edge_data->dtype);
  }

  gdf_size_type nE = i_graph->edgeList->src_indices->size;

  //If an edge satisfies k-core conditions i.e. core_num[src] and core_num[dst]
  //are both greater than or equal to k, copy it to the output graph
  if (hasData) {
    auto inEdge = thrust::make_zip_iterator(thrust::make_tuple(
          thrust::device_pointer_cast(i_src),
          thrust::device_pointer_cast(i_dst),
          thrust::device_pointer_cast(i_wgt)));
    auto outEdge = thrust::make_zip_iterator(thrust::make_tuple(
          thrust::device_pointer_cast(o_src),
          thrust::device_pointer_cast(o_dst),
          thrust::device_pointer_cast(o_wgt)));
    auto ptr = thrust::copy_if(rmm::exec_policy(stream)->on(stream),
        inEdge, inEdge + nE,
        outEdge,
        FilterEdges(k, c_ptr));
    if ((ptr - outEdge) != filteredEdgeCount) { CUGRAPH_FAIL("Edge extraction failed"); }
  } else {
    auto inEdge = thrust::make_zip_iterator(thrust::make_tuple(
          thrust::device_pointer_cast(i_src),
          thrust::device_pointer_cast(i_dst)));
    auto outEdge = thrust::make_zip_iterator(thrust::make_tuple(
          thrust::device_pointer_cast(o_src),
          thrust::device_pointer_cast(o_dst)));
    auto ptr = thrust::copy_if(rmm::exec_policy(stream)->on(stream),
        inEdge, inEdge + nE,
        outEdge,
        FilterEdges(k, c_ptr));
    if ((ptr - outEdge) != filteredEdgeCount) { CUGRAPH_FAIL("Edge extraction failed"); }
  }
  
}

} //namespace

//Extract a subgraph from in_graph (with or without weights)
//to out_graph based on whether edges in in_graph satisfy kcore
//conditions.
//i.e. All edges (s,d,w) in in_graph are copied over to out_graph
//if core_num[s] and core_num[d] are greater than or equal to k.
void extract_subgraph(Graph *in_graph,
                           Graph *out_graph,
                           int * vid,
                           int * core_num,
                           int k,
                           gdf_size_type len,
                           gdf_size_type nV) {
  cudaStream_t stream{nullptr};

  rmm::device_vector<int> c;
  thrust::device_ptr<int> c_ptr = thrust::device_pointer_cast(core_num);
  //We cannot assume that the user provided core numbers per vertex will be in
  //order. Therefore, they need to be reordered by the vertex ids in a temporary
  //array.
  c.resize(nV, 0);
  thrust::device_ptr<int> v_ptr = thrust::device_pointer_cast(vid);
  thrust::scatter(rmm::exec_policy(stream)->on(stream),
      c_ptr, c_ptr + len,
      v_ptr, c.begin());
  c_ptr = thrust::device_pointer_cast(c.data().get());

  cugraph::add_edge_list(in_graph);
  thrust::device_ptr<int> src =
    thrust::device_pointer_cast(static_cast<int*>(in_graph->edgeList->src_indices->data));
  thrust::device_ptr<int> dst =
    thrust::device_pointer_cast(static_cast<int*>(in_graph->edgeList->dest_indices->data));

  //Count number of edges in the input graph that satisfy kcore conditions
  //i.e. core_num[src] and core_num[dst] are both greater than or equal to k
  gdf_size_type nE = in_graph->edgeList->src_indices->size;
  auto edge = thrust::make_zip_iterator(thrust::make_tuple(src, dst));
  int filteredEdgeCount = thrust::count_if(rmm::exec_policy(stream)->on(stream),
      edge, edge + nE, detail::FilterEdges(k, c_ptr));

  //Extract the relevant edges that have satisfied k-core conditions and put them in the output graph
  if (in_graph->edgeList->edge_data != nullptr) {
    switch (in_graph->edgeList->edge_data->dtype) {
      case GDF_FLOAT32:   return detail::extract_edges<float> (in_graph, out_graph, c_ptr, k, filteredEdgeCount);
      case GDF_FLOAT64:   return detail::extract_edges<double>(in_graph, out_graph, c_ptr, k, filteredEdgeCount);
      default: CUGRAPH_FAIL("Unsupported data type");
    }
  }
  else {
    return detail::extract_edges<float> (in_graph, out_graph, c_ptr, k, filteredEdgeCount);
  }
}

void core_number(Graph *graph,
                gdf_column *core_number) {

  CUGRAPH_EXPECTS(graph->adjList != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(graph->adjList->offsets->dtype == GDF_INT32, "Unsupported data type");
  CUGRAPH_EXPECTS(graph->adjList->indices->dtype == GDF_INT32, "Unsupported data type");
  CUGRAPH_EXPECTS(core_number->dtype == GDF_INT32, "Unsupported data type");
  CUGRAPH_EXPECTS(core_number->size == graph->numberOfVertices, "Column size mismatch");

  return detail::core_number_impl(graph, static_cast<int*>(core_number->data));
}

void k_core(Graph *in_graph,
                     int k,
                     gdf_column *vertex_id,
                     gdf_column *core_number,
                     Graph *out_graph) {

  CUGRAPH_EXPECTS(out_graph != nullptr && in_graph != nullptr, "Invalid API parameter");
  gdf_size_type nV = in_graph->numberOfVertices;
  CUGRAPH_EXPECTS(in_graph->adjList->offsets->dtype == GDF_INT32, "Unsupported data type");
  CUGRAPH_EXPECTS(in_graph->adjList->indices->dtype == GDF_INT32, "Unsupported data type");
  CUGRAPH_EXPECTS((vertex_id != nullptr) && (core_number != nullptr), "Invalid API parameter");
  CUGRAPH_EXPECTS(vertex_id->dtype == GDF_INT32, "Unsupported data type");
  CUGRAPH_EXPECTS(core_number->dtype == GDF_INT32, "Unsupported data type");
  CUGRAPH_EXPECTS(core_number->size == vertex_id->size, "Invalid API parameter");
  CUGRAPH_EXPECTS(core_number->size == nV, "Invalid API parameter");
  CUGRAPH_EXPECTS(k >= 0, "Invalid API parameter");

  int * vertex_identifier_ptr = static_cast<int*>(vertex_id->data);
  int * core_number_ptr = static_cast<int*>(core_number->data);
  gdf_size_type vLen = vertex_id->size;

  extract_subgraph(in_graph, out_graph,
      vertex_identifier_ptr, core_number_ptr,
      k, vLen, nV);
}

} //namespace cugraph