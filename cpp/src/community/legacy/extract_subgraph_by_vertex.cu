/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cugraph/algorithms.hpp>
#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/util/device_atomics.cuh>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace {

template <typename vertex_t, typename edge_t, typename weight_t, bool has_weight>
std::unique_ptr<cugraph::legacy::GraphCOO<vertex_t, edge_t, weight_t>> extract_subgraph_by_vertices(
  cugraph::legacy::GraphCOOView<vertex_t, edge_t, weight_t> const& graph,
  vertex_t const* vertices,
  vertex_t num_vertices,
  cudaStream_t stream)
{
  edge_t graph_num_verts = graph.number_of_vertices;

  rmm::device_vector<int64_t> error_count_v{1, 0};
  rmm::device_vector<vertex_t> vertex_used_v{graph_num_verts, num_vertices};

  vertex_t* d_vertex_used = vertex_used_v.data().get();
  int64_t* d_error_count  = error_count_v.data().get();

  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<vertex_t>(0),
    thrust::make_counting_iterator<vertex_t>(num_vertices),
    [vertices, d_vertex_used, d_error_count, graph_num_verts] __device__(vertex_t idx) {
      vertex_t v = vertices[idx];
      if ((v >= 0) && (v < graph_num_verts)) {
        d_vertex_used[v] = idx;
      } else {
        atomicAdd(d_error_count, int64_t{1});
      }
    });

  CUGRAPH_EXPECTS(error_count_v[0] == 0,
                  "Input error... vertices specifies vertex id out of range");

  vertex_t* graph_src    = graph.src_indices;
  vertex_t* graph_dst    = graph.dst_indices;
  weight_t* graph_weight = graph.edge_data;

  // iterate over the edges and count how many make it into the output
  int64_t count = thrust::count_if(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<edge_t>(0),
    thrust::make_counting_iterator<edge_t>(graph.number_of_edges),
    [graph_src, graph_dst, d_vertex_used, num_vertices] __device__(edge_t e) {
      vertex_t s = graph_src[e];
      vertex_t d = graph_dst[e];
      return ((d_vertex_used[s] < num_vertices) && (d_vertex_used[d] < num_vertices));
    });

  if (count > 0) {
    auto result = std::make_unique<cugraph::legacy::GraphCOO<vertex_t, edge_t, weight_t>>(
      num_vertices, count, has_weight);

    vertex_t* d_new_src    = result->src_indices();
    vertex_t* d_new_dst    = result->dst_indices();
    weight_t* d_new_weight = result->edge_data();

    //  reusing error_count as a vertex counter...
    thrust::for_each(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<edge_t>(0),
                     thrust::make_counting_iterator<edge_t>(graph.number_of_edges),
                     [graph_src,
                      graph_dst,
                      graph_weight,
                      d_vertex_used,
                      num_vertices,
                      d_error_count,
                      d_new_src,
                      d_new_dst,
                      d_new_weight] __device__(edge_t e) {
                       vertex_t s = graph_src[e];
                       vertex_t d = graph_dst[e];
                       if ((d_vertex_used[s] < num_vertices) && (d_vertex_used[d] < num_vertices)) {
                         //  NOTE: Could avoid atomic here by doing a inclusive sum, but that would
                         //     require 2*|E| temporary memory.  If this becomes important perhaps
                         //     we make 2 implementations and pick one based on the number of
                         //     vertices in the subgraph set.
                         auto pos       = atomicAdd(d_error_count, int64_t{1});
                         d_new_src[pos] = d_vertex_used[s];
                         d_new_dst[pos] = d_vertex_used[d];
                         if (has_weight) d_new_weight[pos] = graph_weight[e];
                       }
                     });

    return result;
  } else {
    return std::make_unique<cugraph::legacy::GraphCOO<vertex_t, edge_t, weight_t>>(
      0, 0, has_weight);
  }
}
}  // namespace

namespace cugraph {
namespace subgraph {

template <typename VT, typename ET, typename WT>
std::unique_ptr<legacy::GraphCOO<VT, ET, WT>> extract_subgraph_vertex(
  legacy::GraphCOOView<VT, ET, WT> const& graph, VT const* vertices, VT num_vertices)
{
  CUGRAPH_EXPECTS(vertices != nullptr, "Invalid input argument: vertices must be non null");

  cudaStream_t stream{0};

  if (graph.edge_data == nullptr) {
    return extract_subgraph_by_vertices<VT, ET, WT, false>(graph, vertices, num_vertices, stream);
  } else {
    return extract_subgraph_by_vertices<VT, ET, WT, true>(graph, vertices, num_vertices, stream);
  }
}

template std::unique_ptr<legacy::GraphCOO<int32_t, int32_t, float>>
extract_subgraph_vertex<int32_t, int32_t, float>(
  legacy::GraphCOOView<int32_t, int32_t, float> const&, int32_t const*, int32_t);
template std::unique_ptr<legacy::GraphCOO<int32_t, int32_t, double>>
extract_subgraph_vertex<int32_t, int32_t, double>(
  legacy::GraphCOOView<int32_t, int32_t, double> const&, int32_t const*, int32_t);

}  // namespace subgraph
}  // namespace cugraph
