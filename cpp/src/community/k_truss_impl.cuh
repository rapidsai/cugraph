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
#include <chrono>
using namespace std::chrono;

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
        false);

    modified_graph_view = (*modified_graph).view();
  }

  // 2. Find (k-1)-core and exclude edges that do not belong to (k-1)-core
  //#if 0
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
        false);

    modified_graph_view = (*modified_graph).view();

    //undirected_graph_view = (*modified_graph).view();

    if (renumber_map) {  // collapse renumber_map
      unrenumber_int_vertices<vertex_t, multi_gpu>(handle,
                                                   (*tmp_renumber_map).data(),
                                                   (*tmp_renumber_map).size(),
                                                   (*renumber_map).data(),
                                                   *vertex_partition_range_lasts);
    }

    renumber_map = std::move(tmp_renumber_map);
  }
  //#endif

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
        // If renumber is set to True, cur_graph_view and graph_view don't have the same
        // renumbering scheme. Will need to renumber before performing certain operations on
        // graph_view like nbr_intersection.
        false);

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

    auto cur_undirected_graph_view = undirected_graph_view ? *undirected_graph_view : graph_view;
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;

   auto edge_triangle_counts =
      edge_triangle_count<vertex_t, edge_t, multi_gpu>(handle, cur_graph_view, false);

    // Extract all undirected edges
    auto [edgelist_srcs, edgelist_dsts, edgelist_cnts] =
            extract_transform_e(handle,
                                cur_graph_view,
                                edge_src_dummy_property_t{}.view(),
                                edge_dst_dummy_property_t{}.view(),
                                edge_triangle_counts.view(),
                                // FIXME: Replace by lambda function
                                extract_edges<vertex_t, edge_t>{});
    
    // sort the edges by keys where keys are triangle_counts
    auto edgelist_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());
    auto edgelist_last  = thrust::make_zip_iterator(edgelist_srcs.end(), edgelist_dsts.end());

    // Symmetrize the DODG graph

    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};

    cugraph::graph_t<vertex_t, edge_t, false, multi_gpu> cur_graph(handle);
    cur_graph = std::move(*modified_graph);

    std::optional<
      cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                               weight_t>>
      sg_edge_weights{std::nullopt};
    
    edge_weight_view =
      edge_weight ? std::make_optional((*edge_weight).view())
                  : std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt};
  
    std::tie(*modified_graph, std::ignore, std::ignore) = 
      cugraph::symmetrize_graph<vertex_t, edge_t, weight_t, false, multi_gpu>(
        handle,
        std::move(cur_graph),
        std::move(sg_edge_weights),
        //std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
        std::optional<rmm::device_uvector<vertex_t>>{std::nullopt},
        /*
        tmp_renumber_map ? std::optional<rmm::device_uvector<vertex_t>>(std::move(*tmp_renumber_map))
                        : std::nullopt,
        */
        false);
    
    cur_undirected_graph_view = (*modified_graph).view();

    // Sort once
    thrust::sort_by_key(
      handle.get_thrust_policy(),
      edgelist_first,
      edgelist_last,
      edgelist_cnts.begin()
    );

    // FIXME: edgelist_cnts - rename to num_triangles
    auto edge_triangle_count_pair_first =
      thrust::make_zip_iterator(edgelist_first, edgelist_cnts.begin());

    cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edges_to_mask(handle);
    cugraph::edge_property_t<decltype(cur_undirected_graph_view), bool> edge_mask_undirected_graph(handle, cur_undirected_graph_view);
    cugraph::fill_edge_property(handle, cur_undirected_graph_view, edge_mask_undirected_graph.mutable_view(), bool{true});

    size_t prev_chunk_size         = 0;

    auto iteration = -1;

    std::chrono::seconds s (0);
    //std::chrono::duration<double, std::milli> k_truss_ms = duration_cast<milliseconds> (s);
    
    std::chrono::duration<double, std::micro> k_truss_ms = duration_cast<microseconds> (s);
    std::chrono::duration<double, std::micro> intersection_ms = duration_cast<microseconds> (s);
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    auto start = high_resolution_clock::now();

    vertex_t number_edges = 0;
    while (true) {
      iteration += 1;
        
        auto prev_number_of_edges = cur_undirected_graph_view.compute_number_of_edges(handle);

        if (iteration == 2) {
          //break;
        }

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

        // Once identifying the weak edges, perform nbr_intersection on the weak edges.
        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        auto intersection_start = high_resolution_clock::now();
        
        auto [intersection_offsets, intersection_indices] = \
            per_v_pair_dst_nbr_intersection(
                handle,
                cur_undirected_graph_view,
                weak_edgelist_first,
                weak_edgelist_last,
                false);
        
        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        auto intersection_stop = high_resolution_clock::now();

        intersection_ms += duration_cast<microseconds>(intersection_stop - intersection_start);

        


        // Identify (p, q) edges, and form edges (p, q), (p, r) and (q, r)
        // To avoid overcompensation for (q, r) edges, check whether None of the other edges were part of the (p, q) edges.
        // To avoid overcompensation for (p, r) edges, check whether NOne of the other edges were part of the (p, q) and (q, r) edges.

        auto vertex_pair_buffer_p_q =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                      handle.get_stream());

        thrust::tabulate(
            handle.get_thrust_policy(),
            get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
            get_dataframe_buffer_end(vertex_pair_buffer_p_q),
            generate_p_q<vertex_t, edge_t>{
            prev_chunk_size,
            raft::device_span<size_t const>(intersection_offsets.data(),
                                            intersection_offsets.size()),
            raft::device_span<vertex_t const>(intersection_indices.data(),
                                                intersection_indices.size()),
            raft::device_span<vertex_t const>(edgelist_srcs.data() + num_valid_edges,
                                              num_weak_edges),
            raft::device_span<vertex_t const>(edgelist_dsts.data() + num_valid_edges,
                                              num_weak_edges)
            });

        // From nbr_intersection on the undirected graph, we know the endpoints (vertices) of the triangles however
        // we don't know the edges directions. Since edges of the DODG are directed, we can easily recover the
        // direction of the edges with a binary search

        auto vertex_pair_buffer_p_r_edge_p_q =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                          handle.get_stream());
        thrust::tabulate(
            handle.get_thrust_policy(),
            get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_p_q),
            get_dataframe_buffer_end(vertex_pair_buffer_p_r_edge_p_q),
            generate_p_r_or_q_r_from_p_q<vertex_t, edge_t, decltype(edgelist_first), true>{
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
            edgelist_last});
        
        auto vertex_pair_buffer_q_r_edge_p_q =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                          handle.get_stream());

        thrust::tabulate(
            handle.get_thrust_policy(),
            get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q),
            get_dataframe_buffer_end(vertex_pair_buffer_q_r_edge_p_q),
            generate_p_r_or_q_r_from_p_q<vertex_t, edge_t, decltype(edgelist_first), false>{
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
            edgelist_last});       

        auto vertex_pair_buffer_p_q_first = thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_p_q).begin(), std::get<1>(vertex_pair_buffer_p_q).begin());
        auto vertex_pair_buffer_p_r_edge_p_q_first = thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_p_r_edge_p_q).begin(), std::get<1>(vertex_pair_buffer_p_r_edge_p_q).begin());
        auto vertex_pair_buffer_q_r_edge_p_q_first = thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_q_r_edge_p_q).begin(), std::get<1>(vertex_pair_buffer_q_r_edge_p_q).begin());
        
        
        auto triangles_first = thrust::make_zip_iterator(vertex_pair_buffer_p_q_first, vertex_pair_buffer_p_r_edge_p_q_first, vertex_pair_buffer_q_r_edge_p_q_first); // FIXME: not really a triangle but two edges of a triangle so rename
        auto num_triangles = intersection_indices.size();

        auto vertex_pair_buffer_p_q_=
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                          handle.get_stream()); 
        auto vertex_pair_buffer_p_r_edge_p_q_ =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                          handle.get_stream());
        auto vertex_pair_buffer_q_r_edge_p_q_ =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                          handle.get_stream());
        
        auto vertex_pair_buffer_p_q_first_ = thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_p_q_).begin(), std::get<1>(vertex_pair_buffer_p_q_).begin());
        auto vertex_pair_buffer_p_r_edge_p_q_first_ = thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_p_r_edge_p_q_).begin(), std::get<1>(vertex_pair_buffer_p_r_edge_p_q_).begin());
        auto vertex_pair_buffer_q_r_edge_p_q_first_ = thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_q_r_edge_p_q_).begin(), std::get<1>(vertex_pair_buffer_q_r_edge_p_q_).begin());
        auto triangles_first_ = thrust::make_zip_iterator(vertex_pair_buffer_p_q_first_, vertex_pair_buffer_p_r_edge_p_q_first_, vertex_pair_buffer_q_r_edge_p_q_first_); // FIXME: not really a triangle but two edges of a triangle so rename
        
         // Reorder edges' position in the triangle
         thrust::transform(
            handle.get_thrust_policy(),
            triangles_first,
            triangles_first + num_triangles,
            triangles_first_,
            [] __device__(auto triangle) {
                auto edge_p_q = thrust::get<0>(triangle);
                auto edge_p_r = thrust::get<1>(triangle);
                auto edge_q_r = thrust::get<2>(triangle);
                if (thrust::get<1>(edge_p_q) == thrust::get<1>(edge_q_r)) {
                  if (thrust::get<0>(edge_p_q) == thrust::get<0>(edge_p_r)) {
                    return thrust::tie(edge_p_r, edge_q_r, edge_p_q);
                  } else {
                    return thrust::tie(edge_p_r, edge_p_q, edge_q_r);
                  }              
                } else if (thrust::get<1>(edge_p_q) == thrust::get<0>(edge_q_r)) {
                  return thrust::tie(edge_p_q, edge_q_r, edge_p_r);
                
                } else { // Only for debugging purposes. Remove after.
                  printf("\ninvalid combination\n");
                }
            });
      
        thrust::sort(
          handle.get_thrust_policy(),
          triangles_first_,
          triangles_first_ + num_triangles);
    
        auto unique_triangle_end =  thrust::unique(
                                        handle.get_thrust_policy(),
                                        triangles_first_,
                                        triangles_first_ + num_triangles);

        auto num_unique_triangles = thrust::distance(triangles_first_, unique_triangle_end);

        resize_dataframe_buffer(vertex_pair_buffer_p_q_, num_unique_triangles, handle.get_stream());
        resize_dataframe_buffer(vertex_pair_buffer_p_r_edge_p_q_, num_unique_triangles, handle.get_stream());
        resize_dataframe_buffer(vertex_pair_buffer_q_r_edge_p_q_, num_unique_triangles, handle.get_stream());

 
        resize_dataframe_buffer(vertex_pair_buffer_p_q_, 3 * num_unique_triangles, handle.get_stream());

        // Copy p_r edges
        thrust::copy(
            handle.get_thrust_policy(),
            get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_p_q_),
            get_dataframe_buffer_end(vertex_pair_buffer_p_r_edge_p_q_),
            get_dataframe_buffer_begin(vertex_pair_buffer_p_q_) + num_unique_triangles);
        
        thrust::copy(
            handle.get_thrust_policy(),
            get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q_),
            get_dataframe_buffer_end(vertex_pair_buffer_q_r_edge_p_q_),
            get_dataframe_buffer_begin(vertex_pair_buffer_p_q_) + (2*num_unique_triangles));
        
        thrust::sort(
          handle.get_thrust_policy(),
          get_dataframe_buffer_begin(vertex_pair_buffer_p_q_),
          get_dataframe_buffer_end(vertex_pair_buffer_p_q_));
        
        auto unique_pair_count = thrust::unique_count(handle.get_thrust_policy(),
                                                      get_dataframe_buffer_begin(vertex_pair_buffer_p_q_),
                                                      get_dataframe_buffer_end(vertex_pair_buffer_p_q_));
        
        auto vertex_pair_buffer_unique = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          unique_pair_count, handle.get_stream());
  
        rmm::device_uvector<edge_t> decrease_count(unique_pair_count, handle.get_stream());

        thrust::reduce_by_key(handle.get_thrust_policy(),
                              get_dataframe_buffer_begin(vertex_pair_buffer_p_q_),
                              get_dataframe_buffer_end(vertex_pair_buffer_p_q_),
                              thrust::make_constant_iterator(size_t{1}),
                              get_dataframe_buffer_begin(vertex_pair_buffer_unique),
                              decrease_count.begin(),
                              thrust::equal_to<thrust::tuple<vertex_t, vertex_t>>{});

      
        // Update the triangle count of edges

        auto weak_srcs = edgelist_srcs.begin() + num_valid_edges;
        auto weak_dsts = edgelist_dsts.begin() + num_valid_edges;

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
                        // FIXMEEE: thrust::find
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
        
        cugraph::transform_e(
            handle,
            cur_undirected_graph_view,
            edges_to_mask,
            cugraph::edge_src_dummy_property_t{}.view(),
            cugraph::edge_dst_dummy_property_t{}.view(),
            cugraph::edge_dummy_property_t{}.view(),
            [iteration] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {
              
              return false;
            },
            edge_mask_undirected_graph.mutable_view(),
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
            cur_undirected_graph_view,
            edges_to_mask,
            cugraph::edge_src_dummy_property_t{}.view(),
            cugraph::edge_dst_dummy_property_t{}.view(),
            cugraph::edge_dummy_property_t{}.view(),
            [iteration] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {

              return false;
            },
            edge_mask_undirected_graph.mutable_view(),
            false);
        
        cur_undirected_graph_view.attach_edge_mask(edge_mask_undirected_graph.view());

        edgelist_srcs.resize(num_valid_edges, handle.get_stream());
        edgelist_dsts.resize(num_valid_edges, handle.get_stream());
        edgelist_cnts.resize(num_valid_edges, handle.get_stream());

        edgelist_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());
        edgelist_last  = thrust::make_zip_iterator(edgelist_srcs.end(), edgelist_dsts.end());


        //number_edges = cur_undirected_graph_view.compute_number_of_edges(handle);
        if (prev_number_of_edges == cur_undirected_graph_view.compute_number_of_edges(handle)) { break; }

    }

    RAFT_CUDA_TRY(cudaDeviceSynchronize()); 
    auto stop = high_resolution_clock::now();
    k_truss_ms = duration_cast<microseconds>(stop - start);

    std::cout << "k_truss took " << k_truss_ms.count() / 1000 << " milliseconds" <<  std::endl;
    std::cout << "intersection took " << intersection_ms.count()/1000 << " milliseconds" << std::endl;
    std::cout << "percentage during intersection = " << ((intersection_ms.count()/1000) / (k_truss_ms.count() / 1000)) * 100 << std::endl;
    std::cout << "The number of edges = " << cur_undirected_graph_view.compute_number_of_edges(handle) << " and the num_iteration = " << iteration << std::endl;

    std::optional<rmm::device_uvector<weight_t>> edgelist_wgts{std::nullopt};
    
    //#if 0
    std::tie(edgelist_srcs, edgelist_dsts, edgelist_wgts, std::ignore, std::ignore) =
      decompress_to_edgelist(
        handle,
        cur_undirected_graph_view,
        std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
        //edge_weight_view ? std::make_optional(*edge_weight_view) : std::nullopt, // support edgeweights
        std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
        std::optional<raft::device_span<vertex_t const>>{std::nullopt}
        /*
        std::make_optional(
          raft::device_span<vertex_t const>((*renumber_map).data(), (*renumber_map).size())) // Update renumbering if it exist. 
        */
        );
  
    std::tie(edgelist_srcs, edgelist_dsts, edgelist_wgts) =
      symmetrize_edgelist<vertex_t, weight_t, false, multi_gpu>(handle,
                                                                std::move(edgelist_srcs),
                                                                std::move(edgelist_dsts),
                                                                std::move(edgelist_wgts),
                                                                false);
  
    return std::make_tuple(
      std::move(edgelist_srcs), std::move(edgelist_dsts), std::move(edgelist_wgts));
  }
}
}  // namespace cugraph
