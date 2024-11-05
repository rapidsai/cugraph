/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "prims/edge_bucket.cuh"
#include "prims/extract_transform_e.cuh"
#include "prims/per_v_pair_dst_nbr_intersection.cuh"
#include "prims/extract_transform_v_frontier_outgoing_e.cuh"
#include "prims/fill_edge_property.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh"
#include "prims/update_edge_src_dst_property.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/collect_comm_wrapper.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/util/integer_utils.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/optional.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugraph {

template <typename vertex_t, typename edge_t>
struct extract_weak_edges {
  edge_t k{};
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(
    vertex_t src, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) const
  {
    return count < k - 2
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
             : thrust::nullopt;
  }
};


template <typename vertex_t, typename edge_t>
struct extract_edges { // FIXME:  ******************************Remove this functor. For testing purposes only*******************
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t, edge_t>> operator()(
    
    auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto count) const
  {
    return thrust::make_tuple(src, dst, count);
  }
};


template <typename vertex_t, typename edge_t>
struct extract_edges_ { // FIXME:  ******************************Remove this functor. For testing purposes only*******************
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(
    
    auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
  {
    return thrust::make_tuple(src, dst);
  }
};



template <typename vertex_t>
struct extract_masked_edges { // FIXME:  ******************************Remove this functor. For testing purposes only*******************
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(
    
    auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto mask) const
  {
    return mask == 0
            ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
            : thrust::nullopt;
  }
};


template <typename vertex_t, typename edge_t, typename EdgeIterator>
struct extract_triangles_from_weak_edges {
  size_t chunk_start{};
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  raft::device_span<vertex_t const> weak_srcs{};
  raft::device_span<vertex_t const> weak_dsts{};
  EdgeIterator edgelist_first{};
  EdgeIterator weak_edgelist_first{};
  EdgeIterator edgelist_last{};


  __device__ thrust::tuple<vertex_t, vertex_t, vertex_t, vertex_t, vertex_t, vertex_t>
  operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);

    auto edge_p_q = thrust::make_tuple(weak_srcs[chunk_start + idx], weak_dsts[chunk_start + idx]);

    // Extract (p, r) edges
    auto edge_p_r = thrust::make_tuple(weak_srcs[chunk_start + idx], intersection_indices[i]);

    // check for edge existance in the DODG edgelist- FIXME: Create a function
    // Check in the valid edge range
    auto has_edge = thrust::binary_search(
          thrust::seq, edgelist_first, weak_edgelist_first, edge_p_r);
    
    if (!has_edge) { // FIXME: Do binary search instead
      // Search in the weak edge partition.
      has_edge = thrust::binary_search(
          thrust::seq, weak_edgelist_first, edgelist_last, edge_p_r);

      if (!has_edge) {
        // Edge must be in the other direction
        edge_p_r = thrust::make_tuple(thrust::get<1>(edge_p_r), thrust::get<0>(edge_p_r));
      }
    }

    // Extract (q, r) edges
    auto edge_q_r = thrust::make_tuple(weak_dsts[chunk_start + idx], intersection_indices[i]);


    // check for edge existance in the DODG edgelist- FIXME: Can be a function
    // Check in the valid edge range
    has_edge = thrust::binary_search(
          thrust::seq, edgelist_first, weak_edgelist_first, edge_q_r);
    
    if (!has_edge) { // FIXME: Do binary search instead
      // Search in the weak edge partition.
      has_edge = thrust::binary_search(
          thrust::seq, weak_edgelist_first, edgelist_last, edge_q_r);

      if (!has_edge) {
        // Edge must be in the other direction
        edge_q_r = thrust::make_tuple(thrust::get<1>(edge_q_r), thrust::get<0>(edge_q_r));
      }
    }

    return thrust::make_tuple(
      thrust::get<0>(edge_p_q), thrust::get<1>(edge_p_q),
      thrust::get<0>(edge_p_r), thrust::get<1>(edge_p_r),
      thrust::get<0>(edge_q_r), thrust::get<1>(edge_q_r));
  }
};


template <typename vertex_t, typename edge_t>
struct generate_p_q {
  size_t chunk_start{};
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  raft::device_span<vertex_t const> weak_srcs{};
  raft::device_span<vertex_t const> weak_dsts{};


  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);

    return thrust::make_tuple(weak_srcs[chunk_start + idx], weak_dsts[chunk_start + idx]);
  }
};

template <typename vertex_t, typename edge_t, typename EdgeIterator, bool generate_p_r>
struct generate_p_r_or_q_r_from_p_q {
  size_t chunk_start{};
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  raft::device_span<vertex_t const> weak_srcs{};
  raft::device_span<vertex_t const> weak_dsts{};
  EdgeIterator edgelist_first{};
  EdgeIterator weak_edgelist_first{};
  EdgeIterator edgelist_last{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);

    auto edge = thrust::make_tuple(weak_dsts[chunk_start + idx], intersection_indices[i]);

    if constexpr (generate_p_r) {
        edge = thrust::make_tuple(weak_srcs[chunk_start + idx], intersection_indices[i]);
    }
    
    // Check in the valid edge range
    auto has_edge = thrust::binary_search(
          thrust::seq, edgelist_first, weak_edgelist_first, edge);
    
    if (!has_edge) { // FIXME: Do binary search instead
      // Search in the weak edge partition.
      has_edge = thrust::binary_search(
          thrust::seq, weak_edgelist_first, edgelist_last, edge);

      if (!has_edge) { // FIXME: Do binary search instead
        edge = thrust::make_tuple(thrust::get<1>(edge), thrust::get<0>(edge)); // Edge must be in the other direction
      }
    }


    return edge;
  }
};

namespace {

template <typename vertex_t>
struct exclude_self_loop_t {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(
    vertex_t src, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
  {
    return src != dst
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
             : thrust::nullopt;
  }
};

template <typename vertex_t, typename weight_t, typename edge_t>
struct extract_low_to_high_degree_weighted_edges_t {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t, weight_t>> operator()(
    vertex_t src, vertex_t dst, edge_t src_out_degree, edge_t dst_out_degree, weight_t wgt) const
  {
    return (src_out_degree < dst_out_degree)
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t, weight_t>>{thrust::make_tuple(
                 src, dst, wgt)}
             : (((src_out_degree == dst_out_degree) &&
                 (src < dst) /* tie-breaking using vertex ID */)
                  ? thrust::optional<
                      thrust::tuple<vertex_t, vertex_t, weight_t>>{thrust::make_tuple(
                      src, dst, wgt)}
                  : thrust::nullopt);
  }
};

template <typename vertex_t, typename edge_t>
struct extract_low_to_high_degree_edges_t {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(vertex_t src,
                                                                            vertex_t dst,
                                                                            edge_t src_out_degree,
                                                                            edge_t dst_out_degree,
                                                                            thrust::nullopt_t) const
  {
    return (src_out_degree < dst_out_degree)
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
             : (((src_out_degree == dst_out_degree) &&
                 (src < dst) /* tie-breaking using vertex ID */)
                  ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src,
                                                                                           dst)}
                  : thrust::nullopt);
  }
};

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
k_truss(raft::handle_t const& handle,
        graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
        std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
        edge_t k,
        bool do_expensive_check)
{
  // 1. Check input arguments.

  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "Invalid input arguments: K-truss currently supports undirected graphs only.");
  CUGRAPH_EXPECTS(!graph_view.is_multigraph(),
                  "Invalid input arguments: K-truss currently does not support multi-graphs.");

  if (do_expensive_check) {
    // nothing to do
  }

  std::optional<graph_t<vertex_t, edge_t, false, multi_gpu>> modified_graph{std::nullopt};
  std::optional<graph_view_t<vertex_t, edge_t, false, multi_gpu>> modified_graph_view{std::nullopt};
  std::optional<graph_view_t<vertex_t, edge_t, false, multi_gpu>> undirected_graph_view{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, weight_t>>
    edge_weight{std::nullopt};
  std::optional<rmm::device_uvector<weight_t>> wgts{std::nullopt};

  // Ideally, leverage the undirected graph derived from k-core
  undirected_graph_view = graph_view;

  if (graph_view.count_self_loops(handle) > edge_t{0}) {
    auto [srcs, dsts] = extract_transform_e(handle,
                                            graph_view,
                                            edge_src_dummy_property_t{}.view(),
                                            edge_dst_dummy_property_t{}.view(),
                                            edge_dummy_property_t{}.view(),
                                            exclude_self_loop_t<vertex_t>{});

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t>(
          handle, std::move(srcs), std::move(dsts), std::nullopt, std::nullopt, std::nullopt);
    }

    std::tie(*modified_graph, std::ignore, std::ignore, std::ignore, renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{true, graph_view.is_multigraph()},
        true);

    modified_graph_view = (*modified_graph).view();
  }

  // 2. Find (k-1)-core and exclude edges that do not belong to (k-1)-core
  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;

    auto vertex_partition_range_lasts =
      renumber_map
        ? std::make_optional<std::vector<vertex_t>>(cur_graph_view.vertex_partition_range_lasts())
        : std::nullopt;

    rmm::device_uvector<edge_t> core_numbers(cur_graph_view.number_of_vertices(),
                                             handle.get_stream());
    core_number(
      handle, cur_graph_view, core_numbers.data(), k_core_degree_type_t::OUT, size_t{2}, size_t{2});

    raft::device_span<edge_t const> core_number_span{core_numbers.data(), core_numbers.size()};

    auto [srcs, dsts, wgts] = k_core(handle,
                                     cur_graph_view,
                                     edge_weight_view,
                                     k - 1,
                                     std::make_optional(k_core_degree_type_t::OUT),
                                     std::make_optional(core_number_span));

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, wgts, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t>(
          handle, std::move(srcs), std::move(dsts), std::move(wgts), std::nullopt, std::nullopt);
    }

    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};
    std::tie(*modified_graph, edge_weight, std::ignore, std::ignore, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::move(wgts),
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{true, graph_view.is_multigraph()},
        true);

    modified_graph_view = (*modified_graph).view();

    if (renumber_map) {  // collapse renumber_map
      unrenumber_int_vertices<vertex_t, multi_gpu>(handle,
                                                   (*tmp_renumber_map).data(),
                                                   (*tmp_renumber_map).size(),
                                                   (*renumber_map).data(),
                                                   *vertex_partition_range_lasts);
    }

    renumber_map = std::move(tmp_renumber_map);
  }

  // 3. Keep only the edges from a low-degree vertex to a high-degree vertex.

  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;

    auto vertex_partition_range_lasts =
      renumber_map
        ? std::make_optional<std::vector<vertex_t>>(cur_graph_view.vertex_partition_range_lasts())
        : std::nullopt;

    auto out_degrees = cur_graph_view.compute_out_degrees(handle);
    edge_src_property_t<decltype(cur_graph_view), edge_t> edge_src_out_degrees(handle,
                                                                               cur_graph_view);
    edge_dst_property_t<decltype(cur_graph_view), edge_t> edge_dst_out_degrees(handle,
                                                                               cur_graph_view);
    update_edge_src_property(
      handle, cur_graph_view, out_degrees.begin(), edge_src_out_degrees.mutable_view());
    update_edge_dst_property(
      handle, cur_graph_view, out_degrees.begin(), edge_dst_out_degrees.mutable_view());

    rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());

    edge_weight_view =
      edge_weight ? std::make_optional((*edge_weight).view())
                  : std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt};
    if (edge_weight_view) {
      std::tie(srcs, dsts, wgts) = extract_transform_e(
        handle,
        cur_graph_view,
        edge_src_out_degrees.view(),
        edge_dst_out_degrees.view(),
        *edge_weight_view,
        extract_low_to_high_degree_weighted_edges_t<vertex_t, weight_t, edge_t>{});
    } else {
      std::tie(srcs, dsts) =
        extract_transform_e(handle,
                            cur_graph_view,
                            edge_src_out_degrees.view(),
                            edge_dst_out_degrees.view(),
                            edge_dummy_property_t{}.view(),
                            extract_low_to_high_degree_edges_t<vertex_t, edge_t>{});
    }

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, wgts, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t>(
          handle, std::move(srcs), std::move(dsts), std::move(wgts), std::nullopt, std::nullopt);
    }

    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};

    std::tie(*modified_graph, edge_weight, std::ignore, std::ignore, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::move(wgts),
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{false /* now asymmetric */, cur_graph_view.is_multigraph()},
        true);

    modified_graph_view = (*modified_graph).view();
    if (renumber_map) {  // collapse renumber_map
      unrenumber_int_vertices<vertex_t, multi_gpu>(handle,
                                                   (*tmp_renumber_map).data(),
                                                   (*tmp_renumber_map).size(),
                                                   (*renumber_map).data(),
                                                   *vertex_partition_range_lasts);
    }
    renumber_map = std::move(tmp_renumber_map);
  }

  // 4. Compute triangle count using nbr_intersection and unroll weak edges

  {

    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;

   auto edge_triangle_counts =
      edge_triangle_count<vertex_t, edge_t, multi_gpu>(handle, cur_graph_view, false);

    // Extract all directed edges with their count
    // Note. Maintaining this data-structure is not that expensive after applying
    // k-core and DODG however, it would be more efficient to maintain and operate on a
    // graph_view intead.
    auto [edgelist_srcs, edgelist_dsts, edgelist_cnts] =
            extract_transform_e(handle,
                                cur_graph_view,
                                edge_src_dummy_property_t{}.view(),
                                edge_dst_dummy_property_t{}.view(),
                                edge_triangle_counts.view(),
                                // FIXME: Replace by lambda function
                                extract_edges<vertex_t, edge_t>{});
    
    auto edgelist_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());
    auto edgelist_last  = thrust::make_zip_iterator(edgelist_srcs.end(), edgelist_dsts.end());

    // Symmetrize the DODG graph

    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};

    cugraph::graph_t<vertex_t, edge_t, false, multi_gpu> cur_graph(handle);
    cur_graph = std::move(*modified_graph);

    std::tie(*modified_graph, edge_weight, tmp_renumber_map) = 
      cugraph::symmetrize_graph<vertex_t, edge_t, weight_t, false, multi_gpu>(
        handle,
        std::move(cur_graph),
        std::move(edge_weight),
        std::move(renumber_map),
        false);
    
    edge_weight_view =
      edge_weight ? std::make_optional((*edge_weight).view())
                  : std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt};
    renumber_map = std::move(tmp_renumber_map);
    
    // Leverage the undirected graph view to find triangles
    cur_graph_view = (*modified_graph).view();

    // sort the edges by keys once where keys are triangle_counts
    thrust::sort_by_key(
      handle.get_thrust_policy(),
      edgelist_first,
      edgelist_last,
      edgelist_cnts.begin() // FIXME: edgelist_cnts - rename to num_triangles
    );

    auto edge_triangle_count_pair_first =
      thrust::make_zip_iterator(edgelist_first, edgelist_cnts.begin());

    cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edges_to_mask(handle);
    cugraph::edge_property_t<decltype(cur_graph_view), bool> weak_edges_mask(handle, cur_graph_view);
    cugraph::fill_edge_property(handle, cur_graph_view, weak_edges_mask.mutable_view(), bool{true});
    cur_graph_view.attach_edge_mask(weak_edges_mask.view());

    size_t prev_chunk_size         = 0; // FIXME: Add support for chunking

    while (true) {

        auto prev_number_of_edges = cur_graph_view.compute_number_of_edges(handle);

        auto weak_edge_triangle_count_first =
        thrust::stable_partition(handle.get_thrust_policy(),
                                 edge_triangle_count_pair_first,
                                 edge_triangle_count_pair_first + edgelist_srcs.size(),
                                 [k] __device__(auto e) {
                                   auto num_triangles = thrust::get<1>(e);
                                   return num_triangles >= k - 2;
                                 });

        auto num_weak_edges = static_cast<size_t>(
            thrust::distance(weak_edge_triangle_count_first,
                             edge_triangle_count_pair_first + edgelist_srcs.size()));
        
        auto num_valid_edges = edgelist_srcs.size() - num_weak_edges;

        auto weak_edgelist_first = edgelist_first + num_valid_edges;
        auto weak_edgelist_last = edgelist_first + edgelist_srcs.size();
        
        // Perform nbr_intersection of the weak edges leveraging the undirected
        // graph view
        auto [intersection_offsets, intersection_indices] = \
            per_v_pair_dst_nbr_intersection(
                handle,
                cur_graph_view,
                weak_edgelist_first,
                weak_edgelist_last,
                false);

        // Identify (p, q) edges, and form edges (p, q), (p, r) and (q, r)
        // 'triangles_from_weak_edges' contains the triplet pair as follow (p, q, p, r, q, r)
        auto triangles_from_weak_edges = 
          allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t, vertex_t, vertex_t, vertex_t, vertex_t>>(
            intersection_indices.size(),
            handle.get_stream());
    
        // Extract triangle from weak edges
        thrust::tabulate(
          handle.get_thrust_policy(),
          get_dataframe_buffer_begin(triangles_from_weak_edges),
          get_dataframe_buffer_end(triangles_from_weak_edges),
          extract_triangles_from_weak_edges<vertex_t, edge_t, decltype(edgelist_first)>{
            prev_chunk_size,
            raft::device_span<size_t const>(intersection_offsets.data(),
                                            intersection_offsets.size()),
            raft::device_span<vertex_t const>(intersection_indices.data(),
                                                intersection_indices.size()),
            raft::device_span<vertex_t const>(edgelist_srcs.data() + num_valid_edges,
                                              num_weak_edges),
            raft::device_span<vertex_t const>(edgelist_dsts.data() + num_valid_edges,
                                              num_weak_edges),
            edgelist_first,
            edgelist_first + num_valid_edges,
            edgelist_last
          }
        );

        // Reorder each triangle's edges to match the unique order (p, q), (q, r) and (p, r)
        thrust::transform(
          handle.get_thrust_policy(),
          get_dataframe_buffer_begin(triangles_from_weak_edges),
          get_dataframe_buffer_end(triangles_from_weak_edges),
          get_dataframe_buffer_begin(triangles_from_weak_edges),
          [] __device__(auto triangle) {
            auto edge_p_q = thrust::make_tuple(thrust::get<0>(triangle), thrust::get<1>(triangle));
            auto edge_p_r = thrust::make_tuple(thrust::get<2>(triangle), thrust::get<3>(triangle));
            auto edge_q_r = thrust::make_tuple(thrust::get<4>(triangle), thrust::get<5>(triangle));

            if (thrust::get<1>(edge_p_q) == thrust::get<1>(edge_q_r)) {
                  if (thrust::get<0>(edge_p_q) == thrust::get<0>(edge_p_r)) {
                    triangle = thrust::make_tuple(
                      thrust::get<0>(edge_p_r), thrust::get<1>(edge_p_r),
                      thrust::get<0>(edge_q_r), thrust::get<1>(edge_q_r),
                      thrust::get<0>(edge_p_q), thrust::get<1>(edge_p_q)
                    );

                  } else {
                    triangle = thrust::make_tuple(
                      thrust::get<0>(edge_p_r), thrust::get<1>(edge_p_r),
                      thrust::get<0>(edge_p_q), thrust::get<1>(edge_p_q),
                      thrust::get<0>(edge_q_r), thrust::get<1>(edge_q_r)
                    );
                  }              
                } else if (thrust::get<1>(edge_p_q) == thrust::get<0>(edge_q_r)) {
                  triangle = thrust::make_tuple(
                    thrust::get<0>(edge_p_q), thrust::get<1>(edge_p_q),
                    thrust::get<0>(edge_q_r), thrust::get<1>(edge_q_r),
                    thrust::get<0>(edge_p_r), thrust::get<1>(edge_p_r)
                  );
                
                }
                return triangle;
          }
        );

        // Sort and remove duplicated triangles which will lead to overcompensation
        thrust::sort(
          handle.get_thrust_policy(),
          get_dataframe_buffer_begin(triangles_from_weak_edges),
          get_dataframe_buffer_end(triangles_from_weak_edges));
  
        auto unique_triangle_end =  thrust::unique(
                                        handle.get_thrust_policy(),
                                        get_dataframe_buffer_begin(triangles_from_weak_edges),
                                        get_dataframe_buffer_end(triangles_from_weak_edges));

        auto num_unique_triangles = thrust::distance(get_dataframe_buffer_begin(triangles_from_weak_edges), unique_triangle_end);

        resize_dataframe_buffer(triangles_from_weak_edges, num_unique_triangles, handle.get_stream());
        
        auto edgelist_to_update_count =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(3* num_unique_triangles,
                                                                         handle.get_stream()); 
        // Flatten the triangles into an edgelist 
        thrust::transform(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator<edge_t>(0),
          thrust::make_counting_iterator<edge_t>(size_dataframe_buffer(edgelist_to_update_count)),
          get_dataframe_buffer_begin(edgelist_to_update_count),
          [
            num_unique_triangles,
            triangles_from_weak_edges = get_dataframe_buffer_begin(triangles_from_weak_edges)
          ] __device__(auto idx) {
            auto idx_triangle = idx % num_unique_triangles;
            auto idx_vertex_in_triangle = idx / num_unique_triangles;
            auto triangle = (triangles_from_weak_edges + idx_triangle).get_iterator_tuple();

            vertex_t src;
            vertex_t dst;

            if (idx_vertex_in_triangle == 0) {
              src = *(thrust::get<0>(triangle));
              dst = *(thrust::get<1>(triangle));
            }

            if (idx_vertex_in_triangle == 1) {
              src = *(thrust::get<2>(triangle));
              dst = *(thrust::get<3>(triangle));
            }

            if (idx_vertex_in_triangle == 2) {
              src = *(thrust::get<4>(triangle));
              dst = *(thrust::get<5>(triangle));
            }
            
            return thrust::make_tuple(src, dst);
          }
        );

        thrust::sort(
          handle.get_thrust_policy(),
          get_dataframe_buffer_begin(edgelist_to_update_count),
          get_dataframe_buffer_end(edgelist_to_update_count));
        
        auto unique_pair_count = thrust::unique_count(handle.get_thrust_policy(),
                                                      get_dataframe_buffer_begin(edgelist_to_update_count),
                                                      get_dataframe_buffer_end(edgelist_to_update_count));
        
        auto vertex_pair_buffer_unique = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          unique_pair_count, handle.get_stream());
  
        rmm::device_uvector<edge_t> decrease_count(unique_pair_count, handle.get_stream());

        thrust::reduce_by_key(handle.get_thrust_policy(),
                              get_dataframe_buffer_begin(edgelist_to_update_count),
                              get_dataframe_buffer_end(edgelist_to_update_count),
                              thrust::make_constant_iterator(size_t{1}),
                              get_dataframe_buffer_begin(vertex_pair_buffer_unique),
                              decrease_count.begin(),
                              thrust::equal_to<thrust::tuple<vertex_t, vertex_t>>{});
      
        // Update the triangle count of edges in the DODG edgelist
        thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_counting_iterator<edge_t>(0),
                       thrust::make_counting_iterator<edge_t>(unique_pair_count),
                       [
                        vertex_pair_buffer_unique = get_dataframe_buffer_begin(vertex_pair_buffer_unique),
                        decrease_count = decrease_count.begin(),
                        edgelist_cnts = edgelist_cnts.begin(),
                        edgelist_first,
                        weak_edgelist_first = edgelist_first + num_valid_edges,
                        edgelist_last,
                        num_valid_edges
                       ] __device__(auto i) {
                        // Check in the valid edge range
                        auto itr_pair = thrust::lower_bound(
                          thrust::seq, edgelist_first, weak_edgelist_first, vertex_pair_buffer_unique[i]);
                        
                        // Update counts of valid edges only since weak edges will be deleted anyways
                        if ((itr_pair != weak_edgelist_first) && *itr_pair == *(vertex_pair_buffer_unique + i)) {
                          auto idx = thrust::distance(edgelist_first, itr_pair);
                          edgelist_cnts[idx] -= decrease_count[i];
                        }
                       }
                       );
    
        edges_to_mask.clear();
        edges_to_mask.insert(edgelist_srcs.begin() + num_valid_edges,
                             edgelist_srcs.end(),
                             edgelist_dsts.begin() + num_valid_edges);
        
        // Remove weak edges in both direction from the undirected graph view
        cugraph::transform_e(
            handle,
            cur_graph_view,
            edges_to_mask,
            cugraph::edge_src_dummy_property_t{}.view(),
            cugraph::edge_dst_dummy_property_t{}.view(),
            cugraph::edge_dummy_property_t{}.view(),
            [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {
              
              return false;
            },
            weak_edges_mask.mutable_view(),
            false);
        
        edges_to_mask.clear();
        thrust::sort(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(edgelist_dsts.begin() + num_valid_edges, edgelist_srcs.begin() + num_valid_edges),
          thrust::make_zip_iterator(edgelist_dsts.end(), edgelist_srcs.end())
        );
        
        edges_to_mask.insert(edgelist_dsts.begin() + num_valid_edges,
                             edgelist_dsts.end(),
                             edgelist_srcs.begin() + num_valid_edges);
        
        cugraph::transform_e(
            handle,
            cur_graph_view,
            edges_to_mask,
            cugraph::edge_src_dummy_property_t{}.view(),
            cugraph::edge_dst_dummy_property_t{}.view(),
            cugraph::edge_dummy_property_t{}.view(),
            [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {

              return false;
            },
            weak_edges_mask.mutable_view(),
            false);

        edgelist_srcs.resize(num_valid_edges, handle.get_stream());
        edgelist_dsts.resize(num_valid_edges, handle.get_stream());
        edgelist_cnts.resize(num_valid_edges, handle.get_stream());

        edgelist_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());
        edgelist_last  = thrust::make_zip_iterator(edgelist_srcs.end(), edgelist_dsts.end());

        if (prev_number_of_edges == cur_graph_view.compute_number_of_edges(handle)) { break; }

    }

    std::optional<rmm::device_uvector<weight_t>> edgelist_wgts{std::nullopt};
    
    //#if 0
    std::tie(edgelist_srcs, edgelist_dsts, edgelist_wgts, std::ignore, std::ignore) =
      decompress_to_edgelist(
        handle,
        cur_graph_view,
        //std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
        edge_weight_view ? std::make_optional(*edge_weight_view) : std::nullopt, // support edgeweights
        std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
        //std::optional<raft::device_span<vertex_t const>>{std::nullopt}
        std::make_optional(
          raft::device_span<vertex_t const>((*renumber_map).data(), (*renumber_map).size())) // Update renumbering if it exist. 
        );

    /*
    std::tie(edgelist_srcs, edgelist_dsts, edgelist_wgts) =
      symmetrize_edgelist<vertex_t, weight_t, false, multi_gpu>(handle,
                                                                std::move(edgelist_srcs),
                                                                std::move(edgelist_dsts),
                                                                std::move(edgelist_wgts),
                                                                false);
    */
  
    return std::make_tuple(
      std::move(edgelist_srcs), std::move(edgelist_dsts), std::move(edgelist_wgts));
  }
}
}  // namespace cugraph
