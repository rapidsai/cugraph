/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/prims/extract_if_e.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>

#include <thrust/count.h>

namespace cugraph {

namespace {

template <typename vertex_t>
struct valid_and_in_local_vertex_partition_range_t {
  vertex_t num_vertices{};
  vertex_t local_vertex_partition_range_first{};
  vertex_t local_vertex_partition_range_last{};

  __device__ bool operator()(vertex_t v) const
  {
    return is_valid_vertex(num_vertices, v) && (v >= local_vertex_partition_range_first) ||
           (v < local_vertex_partition_range_last);
  }
};

template <typename vertex_t>
struct is_not_self_loop_t {
  __device__ bool operator()(vertex_t src, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t) const
  {
    return src != dst;
  }
};

template <typename edge_t>
struct is_two_t {
  __device__ bool operator()(edge_t core_number) const { return core_number == edge_t{2}; }
};

template <typename vertex_t>
struct in_two_core_t {
  __device__ bool operator()(vertex_t, vertex_t, bool src_in_two_core, bool dst_in_two_core) const
  {
    return src_in_two_core && dst_in_two_core;
  }
};

template <typename vertex_t, typename edge_t>
struct low_to_high_degree_t {
  __device__ bool operator()(vertex_t src,
                             vertex_t dst,
                             edge_t src_out_degree,
                             edge_t dst_out_degree) const
  {
    return (src_out_degree < dst_out_degree) ? true
                                             : (((src_out_degree == dst_out_degree) &&
                                                 (src < dst) /* tie-breaking using vertex ID */)
                                                  ? true
                                                  : false);
  }
};

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void triangle_counts(raft::handle_t const& handle,
                     graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
                     std::optional<raft::device_span<vertex_t const>> vertices,
                     raft::device_span<edge_t> counts,
                     bool do_expensive_check)
{
  CUGRAPH_FAIL("unimplemented.");

  // 1. Check input arguments.

  CUGRAPH_EXPECTS(
    graph_view.is_symmetric(),
    "Invalid input arguments: triangle_counts currently supports undirected graphs only.");
  if (vertices) {
    CUGRAPH_EXPECTS(counts.size() == (*vertices).size(),
                    "Invalid arguments: counts.size() does not coincide with (*vertices).size().");
  } else {
    CUGRAPH_EXPECTS(counts.size() == (*vertices).size(),
                    "Invalid arguments: counts.size() does not coincide with (*vertices).size().");
  }

  if (do_expensive_check) {
    if (vertices) {
      auto num_invalids = thrust::count_if(handle.get_thrust_policy(),
                                           (*vertices).begin(),
                                           (*vertices).end(),
                                           valid_and_in_local_vertex_partition_range_t<vertex_t>{
                                             graph_view.number_of_vertices(),
                                             graph_view.local_vertex_partition_range_first(),
                                             graph_view.local_vertex_partition_range_last()});

      if constexpr (multi_gpu) {
        auto& comm = handle.get_comms();
        num_invalids =
          host_scalar_allreduce(comm, num_invalids, raft::comms::op_t::SUM, handle.get_stream());
      }
      CUGRAPH_EXPECTS(num_invalids == 0,
                      "Invalid input arguments: invalid vertex IDs in *vertices.");
    }
  }

  // 2. Exclude self-loops (FIXME: better mask-out once we add masking support).

  std::optional<graph_t<vertex_t, edge_t, weight_t, false, multi_gpu>> modified_graph{std::nullopt};
  std::optional<graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu>> modified_graph_view{
    std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

  if (graph_view.count_self_loops() > edge_t{0}) {
    rmm::device_uvector<vertex_t> srcs(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> dsts(size_t{0}, handle.get_stream());
    std::tie(srcs, dsts, std::ignore) = extract_if_e(handle,
                                                     graph_view,
                                                     dummy_property_t<vertex_t>{}.device_view(),
                                                     dummy_property_t<vertex_t>{}.device_view(),
                                                     is_not_self_loop_t<vertex_t>{});

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore) =
        shuffle_edgelist_by_gpu_id(handle, std::move(srcs), std::move(dsts), std::nullopt);
    }

    std::tie(*modified_graph, *renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::nullopt,
        cugraph::graph_properties_t{true, graph_view.is_multigraph()},
        true);

    *modified_graph_view = (*modified_graph).view();
  }

  // 3. Find 2-core and exclude edges that do not belong to 2-core (FIXME: better mask-out once we
  // add masking support).

  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;
    auto vertex_partition_range_lasts =
      renumber_map ? cur_graph_view.vertex_partition_range_lasts() : std::nullopt;

    rmm::device_uvector<edge_t> core_numbers(cur_graph_view.number_of_vertices(),
                                             handle.get_stream());
    core_number(
      handle, cur_graph_view, core_numbers.data(), k_core_degree_type_t::OUT, size_t{2}, size_t{2});

    edge_partition_src_property_t<vertex_t, bool> edge_partition_src_in_two_cores(handle,
                                                                                  cur_graph_view);
    edge_partition_src_property_t<vertex_t, bool> edge_partition_dst_in_two_cores(handle,
                                                                                  cur_graph_view);
    auto in_two_core_first =
      thrust::make_transform_iterator(core_numbers.begin(), is_two_t<edge_t>{});
    update_edge_partition_src_property(
      handle, cur_graph_view, in_two_core_first, edge_partition_src_in_two_cores);
    update_edge_partition_dst_property(
      handle, cur_graph_view, in_two_core_first, edge_partition_dst_in_two_cores);
    rmm::device_uvector<vertex_t> srcs(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> dsts(size_t{0}, handle.get_stream());
    std::tie(srcs, dsts, std::ignore) = extract_if_e(handle,
                                                     cur_graph_view,
                                                     edge_partition_src_in_two_cores.device_view(),
                                                     edge_partition_dst_in_two_cores.device_view(),
                                                     in_two_core_t<vertex_t>{});

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore) =
        shuffle_edgelist_by_gpu_id(handle, std::move(srcs), std::move(dsts), std::nullopt);
    }

    rmm::device_uvector<vertex_t> tmp_renumber_map(size_t{0}, handle.get_stream());
    std::tie(*modified_graph, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::nullopt,
        cugraph::graph_properties_t{true, graph_view.is_multigraph()},
        true);

    *modified_graph_view = (*modified_graph).view();

    if (renumber_map) {  // collapse renumber_map
      unrenumber_int_vertices(handle,
                              tmp_renumber_map.data(),
                              tmp_renumber_map.size(),
                              (*renumber_map).data(),
                              vertex_partition_range_lasts);
    }
    *renumber_map = std::move(tmp_renumber_map);
  }

  // 4. Keep only the edges from a low-degree vertex to a high-degree vertex.

  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;
    auto vertex_partition_range_lasts =
      renumber_map ? cur_graph_view.vertex_partition_range_lasts() : std::nullopt;

    auto out_degrees = cur_graph_view.compute_out_degrees(handle);

    edge_partition_src_property_t<vertex_t, edge_t> edge_partition_src_out_degrees(handle,
                                                                                   cur_graph_view);
    edge_partition_src_property_t<vertex_t, edge_t> edge_partition_dst_out_degrees(handle,
                                                                                   cur_graph_view);
    update_edge_partition_src_property(
      handle, cur_graph_view, out_degrees.begin(), edge_partition_src_out_degrees);
    update_edge_partition_dst_property(
      handle, cur_graph_view, out_degrees.begin(), edge_partition_dst_out_degrees);
    rmm::device_uvector<vertex_t> srcs(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> dsts(size_t{0}, handle.get_stream());
    std::tie(srcs, dsts, std::ignore) = extract_if_e(handle,
                                                     cur_graph_view,
                                                     edge_partition_src_out_degrees.device_view(),
                                                     edge_partition_dst_out_degrees.device_view(),
                                                     low_to_high_degree_t<vertex_t, edge_t>{});

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore) =
        shuffle_edgelist_by_gpu_id(handle, std::move(srcs), std::move(dsts), std::nullopt);
    }

    rmm::device_uvector<vertex_t> tmp_renumber_map(size_t{0}, handle.get_stream());
    std::tie(*modified_graph, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::nullopt,
        cugraph::graph_properties_t{false /* now asymmetric */, graph_view.is_multigraph()},
        true);

    *modified_graph_view = (*modified_graph).view();

    if (renumber_map) {  // collapse renumber_map
      unrenumber_int_vertices(handle,
                              tmp_renumber_map.data(),
                              tmp_renumber_map.size(),
                              (*renumber_map).data(),
                              vertex_partition_range_lasts);
    }
    *renumber_map = std::move(tmp_renumber_map);
  }

  // 5. neighbor intersection

#if 0
  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;

    for_each_nbr_intersection_of_e_endpoints(hanlde, cur_graph_view, dummy_property_t<vertex_t>{}.device_view(),  dummy_property_t<vertex_t>{}.device_view(), []__device__(vertex_t src, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, raft::device_span<vertex_t> intersections) {
  [src] += intersections.size();
  [dst] += intersections.size();
  copy_intersections_to_consec.
});  // for triangle counting, clustering coefficient, K-truss... if just counting the total number of triangles per graph, transform_reduce_nbr_intersection_of_e_endpoints could be more efficient.
   // for_each_nbr_intersection_of_v_pairs() for Jaccard & Overlap coefficients, and Node2Vec

  segmented_copy?
  // delayed copy works only if memory assigned for intersection segments are still valid... This may not be a case here.
  // Add for_each_triangle_of_e_endpoints?  ambiguous for asymmetric graphs.
  }

  // 6. Update triangle counts from neighbor intersections.

  // 7. If vertices.has_value(), gather triangle counts (FIXME: This is inefficient if
  // (*vertices).size() is small, we may better extract a subgraph including only the edges between
  // *vertices and (*vertices)'s one-hop neighbors and count triangles using the subgraph).
#endif

  return;
}

}  // namespace cugraph
