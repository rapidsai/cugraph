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

//template <typename vertex_t>
template <typename GraphViewType, typename vertex_t>
void order_edge_based_on_dodg(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  raft::device_span<vertex_t> edgelist_srcs,
  raft::device_span<vertex_t> edgelist_dsts)

{
  // FIXME: Use global comm for debugging purposes
  // then replace it by minor comm once the accuracy is verified
  auto& comm                 = handle.get_comms();
  auto const comm_size       = comm.get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  auto vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();

  rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                          handle.get_stream());

  raft::update_device(d_vertex_partition_range_lasts.data(),
                  vertex_partition_range_lasts.data(),
                  vertex_partition_range_lasts.size(),
                  handle.get_stream());
  
  auto func = cugraph::detail::compute_gpu_id_from_int_edge_endpoints_t<vertex_t>{
          raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                            d_vertex_partition_range_lasts.size()),
          comm_size,
          major_comm_size,
          minor_comm_size};


  rmm::device_uvector<vertex_t> cp_edgelist_srcs(edgelist_srcs.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> cp_edgelist_dsts(edgelist_srcs.size(), handle.get_stream());

  thrust::copy(
    handle.get_thrust_policy(),
    thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_srcs.begin()),
    thrust::make_zip_iterator(edgelist_dsts.end(), edgelist_dsts.end()),
    thrust::make_zip_iterator(cp_edgelist_srcs.begin(), cp_edgelist_dsts.begin()));

  auto d_tx_counts = cugraph::groupby_and_count(
    thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin()),
    thrust::make_zip_iterator(edgelist_srcs.end(), edgelist_dsts.end()),
    [func]__device__(auto val) {
      return func(val);
  },
  comm_size,
  std::numeric_limits<size_t>::max(),
  handle.get_stream());


  std::vector<size_t> h_tx_counts(d_tx_counts.size());

  raft::update_host(h_tx_counts.data(),
                    d_tx_counts.data(),
                    d_tx_counts.size(),
                    handle.get_stream());
  
  rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
  std::vector<size_t> rx_counts{};

  std::tie(srcs, rx_counts) =
    shuffle_values(handle.get_comms(), cp_edge_srcs_from_triangle.begin(), h_tx_counts, handle.get_stream());
          
  std::tie(dsts, std::ignore) =
    shuffle_values(handle.get_comms(), cp_edge_dsts_from_triangle.begin(), h_tx_counts, handle.get_stream());

  edge_exists = graph_view.has_edge(
                  handle,
                  raft::device_span<vertex_t const>(srcs.data(), srcs.size()),
                  raft::device_span<vertex_t const>(dsts.data(), dsts.size())
                );
  
  std::tie(edge_exists, std::ignore) =
            shuffle_values(handle.get_comms(), edge_exists.begin(), rx_counts, handle.get_stream());
  

  thrust::sort_by_key(
            handle.get_thrust_policy(),
            thrust::make_zip_iterator(
                cp_edgelist_srcs.begin(),
                cp_edgelist_dsts.begin()),
            thrust::make_zip_iterator(
                cp_edgelist_srcs.end(),
                cp_edgelist_dsts.end()),
            edge_exists.begin());
  
  thrust::transform(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator<edge_t>(0),
            thrust::make_counting_iterator<edge_t>(edgelist_srcs.size()),
            edge_exists.begin(),
            [
              //num_unique_triangles,
              edge_exists = edge_exists.data(),
              edgelist_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin()),
              cp_edgelist_first = thrust::make_zip_iterator(cp_edgelist_srcs.begin(), cp_edgelist_dsts.begin()),
              cp_edgelist_last = thrust::make_zip_iterator(cp_edgelist_srcs.end(), cp_edgelist_dsts.end()),
            ] __device__(auto idx) {
              auto src = thrust::get<0>(edgelist_first[idx]);
              auto dst = thrust::get<1>(edgelist_first[idx]);

              auto itr_pair = thrust::lower_bound(
                thrust::seq, cp_edgelist_first, cp_edgelist_last, thrust::make_tuple(src, dst));
              
              auto idx_pair = thrust::distance(cp_edgelist_first, itr_pair);

              return edge_exists[idx_pair];
            }
          );

}

template <typename vertex_t, typename edge_t>
struct extract_weak_edges {
  edge_t k{};
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(
    vertex_t src, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) const
  {
    //printf("\nsrc = %d, dst = %d, count = %d\n", src, dst, count);
    //auto count_ = thrust::get<0>(count);
    return ((count < k - 2) && (count > 0)) 
    //return count < k - 2 // FIXME: might be faster to skip edges with count = 0
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
struct extract_triangles_from_weak_edges {
  size_t chunk_start{};
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  raft::device_span<vertex_t const> weak_srcs{};
  raft::device_span<vertex_t const> weak_dsts{};

  __device__ thrust::tuple<vertex_t, vertex_t, vertex_t, vertex_t, vertex_t, vertex_t>
  operator()(edge_t i) const
  {

    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);

    auto edge_p_q = thrust::make_tuple(weak_srcs[chunk_start + idx], weak_dsts[chunk_start + idx]);

    // Extract (p, r) edges
    auto edge_p_r = thrust::make_tuple(weak_srcs[chunk_start + idx], intersection_indices[i]);

    // Extract (q, r) edges
    auto edge_q_r = thrust::make_tuple(weak_dsts[chunk_start + idx], intersection_indices[i]);

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

  cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edgelist_dodg(handle);

  cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, bool> dodg_mask(handle, graph_view);

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


    cugraph::fill_edge_property(handle, cur_graph_view, dodg_mask.mutable_view(), bool{false});

    // Masking edges not part of the DODG
    edgelist_dodg.insert(srcs.begin(),
                         srcs.end(),
                         dsts.begin());
    
    cugraph::transform_e(
            handle,
            cur_graph_view,
            edgelist_dodg,
            cugraph::edge_src_dummy_property_t{}.view(),
            cugraph::edge_dst_dummy_property_t{}.view(),
            cugraph::edge_dummy_property_t{}.view(),
            [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {
              
              return true;
            },
            dodg_mask.mutable_view(),
            false);
    
    edgelist_dodg.clear();
  }

  // 4. Compute triangle count using nbr_intersection and unroll weak edges

  {

    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;

    cugraph::edge_property_t<decltype(cur_graph_view), bool> weak_edges_mask(handle, cur_graph_view);
    cugraph::fill_edge_property(handle, cur_graph_view, weak_edges_mask.mutable_view(), bool{true});
    
    // Attach mask
    cur_graph_view.attach_edge_mask(dodg_mask.view());

    auto edge_triangle_counts =
        edge_triangle_count<vertex_t, edge_t, multi_gpu>(handle, cur_graph_view, false);

    cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edgelist_weak(handle);
    cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edges_to_decrement_count(handle);

    size_t prev_chunk_size         = 0; // FIXME: Add support for chunking


    while (true) {
        
        cur_graph_view.clear_edge_mask();
        cur_graph_view.attach_edge_mask(dodg_mask.view());
        
        // Extract weak edges
        auto [weak_edgelist_srcs, weak_edgelist_dsts] =
                extract_transform_e(handle,
                                    cur_graph_view,
                                    edge_src_dummy_property_t{}.view(),
                                    edge_dst_dummy_property_t{}.view(),
                                    edge_triangle_counts.view(),
                                    //view_concat(edge_triangle_counts.view(), weak_edges_mask.view()),
                                    // FIXME: Replace by lambda function
                                    extract_weak_edges<vertex_t, edge_t>{k});

        auto weak_edgelist_first = thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin());
        auto weak_edgelist_last  = thrust::make_zip_iterator(weak_edgelist_srcs.end(), weak_edgelist_dsts.end());

        
        // Perform nbr_intersection of the weak edges from the undirected
        // graph view
        cur_graph_view.clear_edge_mask();
        // Attach the weak edge mask
        cur_graph_view.attach_edge_mask(weak_edges_mask.view());


        auto [intersection_offsets, intersection_indices] = \
            per_v_pair_dst_nbr_intersection(
                handle,
                cur_graph_view,
                weak_edgelist_first,
                weak_edgelist_last,
                false);

        // Identify (p, q) edges, and form edges (p, q), (p, r) and (q, r)
        // To avoid overcompensation, redirect all edges in the triangle to follow this unique
        // pattern: (p, q) then (q, r) then (p, r)

        auto triangles_from_weak_edges = 
          allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t, vertex_t, vertex_t, vertex_t, vertex_t>>(
            intersection_indices.size(),
            handle.get_stream());
    
        // Form (p, q) edges
        // Extract triangle from weak
        thrust::tabulate(
          handle.get_thrust_policy(),
          get_dataframe_buffer_begin(triangles_from_weak_edges),
          get_dataframe_buffer_end(triangles_from_weak_edges),
          extract_triangles_from_weak_edges<vertex_t, edge_t>{
            prev_chunk_size,
            raft::device_span<size_t const>(intersection_offsets.data(),
                                            intersection_offsets.size()),
            raft::device_span<vertex_t const>(intersection_indices.data(),
                                                intersection_indices.size()),
            raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
            raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size())
          }
        );

        cur_graph_view.clear_edge_mask();
        // Check for edge existance on the directed graph view
        cur_graph_view.attach_edge_mask(dodg_mask.view());

        rmm::device_uvector<bool> edge_exists(0, handle.get_stream());

        // Handling (p, r) edges

        if constexpr (multi_gpu) {
          // (p, q) edges are owned by the current GPU while (p, r) and (q, r)
          // can be owned by different GPUs
          // Ordering (p, q) edges based on the DODG
          order_edge_based_on_dodg<decltype(cur_graph_view), vertex_t>(
            handle,
            cur_graph_view,
            raft::device_span<vertex_t>(
              std::get<2>(triangles_from_weak_edges).data(),
              std::get<2>(triangles_from_weak_edges).size()),    
            raft::device_span<vertex_t>(
              std::get<3>(triangles_from_weak_edges).data(),
              std::get<3>(triangles_from_weak_edges).size())
            raft::device_span<bool_t>(
              edge_exists.data(),
              edge_exists.size())
          );


          auto vertex_partition_range_lasts = cur_graph_view.vertex_partition_range_lasts();

          rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                                  handle.get_stream());
          

          raft::update_device(d_vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.size(),
                          handle.get_stream());
          
          auto& comm                 = handle.get_comms();
          auto const comm_size       = comm.get_size();
          auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
          auto const major_comm_size = major_comm.get_size();
          auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
          auto const minor_comm_size = minor_comm.get_size();

          auto my_rank = handle.get_comms().get_rank();


          if (my_rank == 1) {
            printf("****initial****\n\n");
            raft::print_device_vector("triangle_srcs", std::get<0>(triangles_from_weak_edges).data(), std::get<0>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("triangle_dsts", std::get<1>(triangles_from_weak_edges).data(), std::get<1>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("triangle_srcs_pr", std::get<2>(triangles_from_weak_edges).data(), std::get<2>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("triangle_dsts_pr", std::get<3>(triangles_from_weak_edges).data(), std::get<3>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("triangle_srcs_qr", std::get<4>(triangles_from_weak_edges).data(), std::get<4>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("triangle_dsts_qr", std::get<5>(triangles_from_weak_edges).data(), std::get<5>(triangles_from_weak_edges).size(), std::cout);
          }


          auto func = cugraph::detail::compute_gpu_id_from_int_edge_endpoints_t<vertex_t>{
          raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                            d_vertex_partition_range_lasts.size()),
          comm_size,
          major_comm_size,
          minor_comm_size};


          rmm::device_uvector<vertex_t> cp_edge_srcs_from_triangle(intersection_indices.size(), handle.get_stream());
          rmm::device_uvector<vertex_t> cp_edge_dsts_from_triangle(intersection_indices.size(), handle.get_stream());

          thrust::copy(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(std::get<2>(triangles_from_weak_edges).begin(), std::get<3>(triangles_from_weak_edges).begin()),
          thrust::make_zip_iterator(std::get<2>(triangles_from_weak_edges).end(), std::get<3>(triangles_from_weak_edges).end()),
          thrust::make_zip_iterator(cp_edge_srcs_from_triangle.begin(), cp_edge_dsts_from_triangle.begin()));

          //std::cout << "part 0" << std::endl;
          auto d_tx_counts = cugraph::groupby_and_count(
          //thrust::make_zip_iterator(std::get<2>(triangles_from_weak_edges).begin(), std::get<3>(triangles_from_weak_edges).begin()),
          //thrust::make_zip_iterator(std::get<2>(triangles_from_weak_edges).end(), std::get<3>(triangles_from_weak_edges).end()),
          thrust::make_zip_iterator(cp_edge_srcs_from_triangle.begin(), cp_edge_dsts_from_triangle.begin()),
          thrust::make_zip_iterator(cp_edge_srcs_from_triangle.end(), cp_edge_dsts_from_triangle.end()),
          [func, comm_size]__device__(auto val) {
          //[func, min_comm_size]__device__(auto val) {
            //printf("\nval = %d\n", val);
            return func(val); //% major_comm_size;
            //return func(val) % minor_comm_size;
          },
          //major_comm_size,
          //minor_comm_size,
          comm_size,
          std::numeric_limits<size_t>::max(),
          handle.get_stream());

          //std::cout << "part 1" << std::endl;

          //printf("\ncount size = %d\n", d_tx_counts.size());
          raft::print_device_vector("d_tx_counts", d_tx_counts.data(), d_tx_counts.size(), std::cout);

          std::vector<size_t> h_tx_counts(d_tx_counts.size());

          raft::update_host(h_tx_counts.data(),
                            d_tx_counts.data(),
                            d_tx_counts.size(),
                            handle.get_stream());
          



          
          raft::print_host_vector("h_tx_counts", h_tx_counts.data(), h_tx_counts.size(), std::cout);
          

          
          rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
          rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
          std::vector<size_t> rx_counts{};

          std::cout << "part 2" << std::endl;
          std::cout << "h_tx_counts size = " << h_tx_counts.size() << " copy egdes size " << cp_edge_srcs_from_triangle.size() << " comm size = " << comm_size << std::endl;

          std::tie(srcs, rx_counts) =
            shuffle_values(handle.get_comms(), cp_edge_srcs_from_triangle.begin(), h_tx_counts, handle.get_stream());
          
          std::tie(dsts, std::ignore) =
            shuffle_values(handle.get_comms(), cp_edge_dsts_from_triangle.begin(), h_tx_counts, handle.get_stream());

          std::cout << "part 3" << std::endl;
          /*
          raft::print_device_vector("srcs", srcs.data(), srcs.size(), std::cout);
          raft::print_device_vector("dsts", dsts.data(), dsts.size(), std::cout);
          */


          edge_exists = cur_graph_view.has_edge(
                          handle,
                          raft::device_span<vertex_t const>(srcs.data(), srcs.size()),
                          raft::device_span<vertex_t const>(dsts.data(), dsts.size())
                        );
          
          std::cout << "part 4" << std::endl;
              
          std::tie(edge_exists, std::ignore) =
            shuffle_values(handle.get_comms(), edge_exists.begin(), rx_counts, handle.get_stream());

          std::cout << "part 5" << std::endl;
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            thrust::make_zip_iterator(
                cp_edge_srcs_from_triangle.begin(),
                cp_edge_dsts_from_triangle.begin()),
            thrust::make_zip_iterator(
                cp_edge_srcs_from_triangle.end(),
                cp_edge_dsts_from_triangle.end()),
            edge_exists.begin() // FIXME: edgelist_cnts - rename to num_triangles
          );
          std::cout << "part 6" << std::endl;
          
          
          // Update count
          /*
          if (my_rank == 1) {
            printf("\nedge existance before updating\n");
            raft::print_device_vector("edge_exists", edge_exists.data(), edge_exists.size(), std::cout);
          }
          */

          thrust::transform(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator<edge_t>(0),
            thrust::make_counting_iterator<edge_t>(intersection_indices.size()),
            edge_exists.begin(),
            [
              //num_unique_triangles,
              edge_exists = edge_exists.data(),
              edge_p_r_first = thrust::make_zip_iterator(
                std::get<2>(triangles_from_weak_edges).begin(),
                std::get<3>(triangles_from_weak_edges).begin()),
              cp_edge_pr_first = thrust::make_zip_iterator(
                cp_edge_srcs_from_triangle.begin(),
                cp_edge_dsts_from_triangle.begin()),
              cp_edge_pr_last = thrust::make_zip_iterator(
                cp_edge_srcs_from_triangle.end(),
                cp_edge_dsts_from_triangle.end()),
              rank = my_rank
            ] __device__(auto idx) {
              auto src = thrust::get<0>(edge_p_r_first[idx]);
              auto dst = thrust::get<1>(edge_p_r_first[idx]);

              auto itr_pair = thrust::lower_bound(
                thrust::seq, cp_edge_pr_first, cp_edge_pr_last, thrust::make_tuple(src, dst));
              
              auto idx_pair = thrust::distance(cp_edge_pr_first, itr_pair);
              if (rank == 1) {
              }

              return edge_exists[idx_pair];
            }
          );

          std::cout << "part 7" << std::endl;

          /*
          if (my_rank == 1) {
            printf("\nedge existance after updating\n");
            raft::print_device_vector("srcs", srcs.data(), srcs.size(), std::cout);
            raft::print_device_vector("dsts", dsts.data(), dsts.size(), std::cout);
            raft::print_device_vector("edge_exists", edge_exists.data(), edge_exists.size(), std::cout);
          }
          */

        } else {

          edge_exists = cur_graph_view.has_edge(
                          handle,
                          raft::device_span<vertex_t const>(std::get<2>(triangles_from_weak_edges).data(), intersection_indices.size()),
                          raft::device_span<vertex_t const>(std::get<3>(triangles_from_weak_edges).data(), intersection_indices.size())
                        );

        }
        //printf("****************p, r*********************\n\n");
        //raft::print_device_vector("edge_exists", edge_exists.data(), edge_exists.size(), std::cout);

        //raft::print_device_vector("edge_exist_1", edge_exists.data(), edge_exists.size(), std::cout);

        // From nbr_intersection on the undirected graph, we know the endpoints (vertices) of the triangles however
        // we don't know the edges directions. Since edges of the DODG are directed, we can easily recover the
        // direction of the edges with a binary search
        
        // Match DODG edges
        thrust::transform(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator<edge_t>(0),
          thrust::make_counting_iterator<edge_t>(intersection_indices.size()),
          thrust::make_zip_iterator(
            std::get<2>(triangles_from_weak_edges).begin(),
            std::get<3>(triangles_from_weak_edges).begin()),
          [
            //num_unique_triangles,
            edge_exists = edge_exists.data(),
            edge_p_r = thrust::make_zip_iterator(
            std::get<2>(triangles_from_weak_edges).begin(),
            std::get<3>(triangles_from_weak_edges).begin())
          ] __device__(auto idx) {
            auto src = thrust::get<0>(edge_p_r[idx]);
            auto dst = thrust::get<1>(edge_p_r[idx]);

            return edge_exists[idx] ? thrust::make_tuple(src, dst) : thrust::make_tuple(dst, src);
          }
        );

        



        // Handling (q, r) edges

        //rmm::device_uvector<bool> edge_exists_(0, handle.get_stream());


        if constexpr (multi_gpu) {

          auto vertex_partition_range_lasts = cur_graph_view.vertex_partition_range_lasts();

          rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                                  handle.get_stream());
          

          raft::update_device(d_vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.size(),
                          handle.get_stream());
          
          auto& comm                 = handle.get_comms();
          auto const comm_size       = comm.get_size();
          auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
          auto const major_comm_size = major_comm.get_size();
          auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
          auto const minor_comm_size = minor_comm.get_size();

          auto my_rank = handle.get_comms().get_rank();


          
          if (my_rank == 1) {
            printf("\nAfter re-ordering (p, r) edges, \n");
            raft::print_device_vector("*triangle_srcs", std::get<0>(triangles_from_weak_edges).data(), std::get<0>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts", std::get<1>(triangles_from_weak_edges).data(), std::get<1>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_srcs_pr", std::get<2>(triangles_from_weak_edges).data(), std::get<2>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts_pr", std::get<3>(triangles_from_weak_edges).data(), std::get<3>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_srcs_qr", std::get<4>(triangles_from_weak_edges).data(), std::get<4>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts_qr", std::get<5>(triangles_from_weak_edges).data(), std::get<5>(triangles_from_weak_edges).size(), std::cout);
          }
          






          auto func = cugraph::detail::compute_gpu_id_from_int_edge_endpoints_t<vertex_t>{
          raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                            d_vertex_partition_range_lasts.size()),
          comm_size,
          major_comm_size,
          minor_comm_size};

          rmm::device_uvector<vertex_t> cp_edge_srcs_from_triangle(intersection_indices.size(), handle.get_stream());
          rmm::device_uvector<vertex_t> cp_edge_dsts_from_triangle(intersection_indices.size(), handle.get_stream());

          thrust::copy(
            handle.get_thrust_policy(),
            thrust::make_zip_iterator(std::get<4>(triangles_from_weak_edges).begin(), std::get<5>(triangles_from_weak_edges).begin()),
            thrust::make_zip_iterator(std::get<4>(triangles_from_weak_edges).end(), std::get<5>(triangles_from_weak_edges).end()),
            thrust::make_zip_iterator(cp_edge_srcs_from_triangle.begin(), cp_edge_dsts_from_triangle.begin()));

          auto d_tx_counts = cugraph::groupby_and_count(
          //thrust::make_zip_iterator(std::get<4>(triangles_from_weak_edges).begin(), std::get<5>(triangles_from_weak_edges).begin()),
          //thrust::make_zip_iterator(std::get<4>(triangles_from_weak_edges).end(), std::get<5>(triangles_from_weak_edges).end()),
          thrust::make_zip_iterator(cp_edge_srcs_from_triangle.begin(), cp_edge_dsts_from_triangle.begin()),
          thrust::make_zip_iterator(cp_edge_srcs_from_triangle.end(), cp_edge_dsts_from_triangle.end()),
          [func, comm_size]__device__(auto val) {
          //[func, min_comm_size]__device__(auto val) {
            //printf("\nval = %d\n", val);
            return func(val); //% major_comm_size;
            //return func(val) % minor_comm_size;
          },
          //major_comm_size,
          //minor_comm_size,
          comm_size,
          std::numeric_limits<size_t>::max(),
          handle.get_stream());

          if (my_rank == 1) {
            printf("\nAfter groupby (q, r)\n");
            raft::print_device_vector("*triangle_srcs", std::get<0>(triangles_from_weak_edges).data(), std::get<0>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts", std::get<1>(triangles_from_weak_edges).data(), std::get<1>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_srcs_pr", std::get<2>(triangles_from_weak_edges).data(), std::get<2>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts_pr", std::get<3>(triangles_from_weak_edges).data(), std::get<3>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_srcs_qr", std::get<4>(triangles_from_weak_edges).data(), std::get<4>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts_qr", std::get<5>(triangles_from_weak_edges).data(), std::get<5>(triangles_from_weak_edges).size(), std::cout);
          }




          //printf("\ncount size = %d\n", d_tx_counts.size());
          //raft::print_device_vector("d_tx_counts", d_tx_counts.data(), d_tx_counts.size(), std::cout);

          std::vector<size_t> h_tx_counts(d_tx_counts.size());

          raft::update_host(h_tx_counts.data(),
                            d_tx_counts.data(),
                            d_tx_counts.size(),
                            handle.get_stream());
          
          rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
          rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
          std::vector<size_t> rx_counts{};

          std::vector<size_t> rx_counts_{};

          std::tie(srcs, rx_counts) =
            shuffle_values(handle.get_comms(), cp_edge_srcs_from_triangle.begin(), h_tx_counts, handle.get_stream());
          
          std::tie(dsts, std::ignore) =
            shuffle_values(handle.get_comms(), cp_edge_dsts_from_triangle.begin(), h_tx_counts, handle.get_stream());

          
          //raft::print_host_vector("rx_counts", rx_counts.data(), rx_counts.size(), std::cout);
          //raft::print_host_vector("rx_counts_", rx_counts_.data(), rx_counts_.size(), std::cout);
          /*
          if (my_rank == 1) {
            raft::print_device_vector("srcs", srcs.data(), srcs.size(), std::cout);
            raft::print_device_vector("dsts", dsts.data(), dsts.size(), std::cout);
          }
          */


          edge_exists = cur_graph_view.has_edge(
                          handle,
                          raft::device_span<vertex_t const>(srcs.data(), srcs.size()),
                          raft::device_span<vertex_t const>(dsts.data(), dsts.size())
                        );
          
          /*
          if (my_rank == 1) {
            raft::print_device_vector("*edge_exists", edge_exists.data(), edge_exists.size(), std::cout);
          }
          */
          std::tie(edge_exists, std::ignore) =
            shuffle_values(handle.get_comms(), edge_exists.begin(), rx_counts, handle.get_stream()); 


          

          thrust::sort_by_key(
            handle.get_thrust_policy(),
            thrust::make_zip_iterator(
                cp_edge_srcs_from_triangle.begin(),
                cp_edge_dsts_from_triangle.begin()),
            thrust::make_zip_iterator(
                cp_edge_srcs_from_triangle.end(),
                cp_edge_dsts_from_triangle.end()),
            edge_exists.begin() // FIXME: edgelist_cnts - rename to num_triangles
          );


          thrust::transform(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator<edge_t>(0),
            thrust::make_counting_iterator<edge_t>(intersection_indices.size()),
            edge_exists.begin(),
            [
              //num_unique_triangles,
              edge_exists = edge_exists.data(),
              edge_p_r_first = thrust::make_zip_iterator(
                std::get<4>(triangles_from_weak_edges).begin(),
                std::get<5>(triangles_from_weak_edges).begin()),
              cp_edge_pr_first = thrust::make_zip_iterator(
                cp_edge_srcs_from_triangle.begin(),
                cp_edge_dsts_from_triangle.begin()),
              cp_edge_pr_last = thrust::make_zip_iterator(
                cp_edge_srcs_from_triangle.end(),
                cp_edge_dsts_from_triangle.end()),
              rank = my_rank
            ] __device__(auto idx) {
              auto src = thrust::get<0>(edge_p_r_first[idx]);
              auto dst = thrust::get<1>(edge_p_r_first[idx]);

              auto itr_pair = thrust::lower_bound(
                thrust::seq, cp_edge_pr_first, cp_edge_pr_last, thrust::make_tuple(src, dst));
              
              auto idx_pair = thrust::distance(cp_edge_pr_first, itr_pair);
              if (rank == 1) {
              }

              return edge_exists[idx_pair];
            }
          );
          
          /*
          if (my_rank == 1) {
            printf("****************q, r*********************\n\n");
            raft::print_device_vector("edge_exists", edge_exists.data(), edge_exists.size(), std::cout);
          }
          */
        
        
        } else {

          edge_exists = cur_graph_view.has_edge(
                          handle,
                          raft::device_span<vertex_t const>(std::get<4>(triangles_from_weak_edges).data(), intersection_indices.size()),
                          raft::device_span<vertex_t const>(std::get<5>(triangles_from_weak_edges).data(), intersection_indices.size())
                        );
        }
        /*
        if (my_rank == 1) {
          printf("****************q, r*********************\n\n");
          raft::print_device_vector("edge_exists_", edge_exists_.data(), edge_exists_.size(), std::cout);
        }
        */

        thrust::transform(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator<edge_t>(0),
          thrust::make_counting_iterator<edge_t>(intersection_indices.size()),
          thrust::make_zip_iterator(
            std::get<4>(triangles_from_weak_edges).begin(),
            std::get<5>(triangles_from_weak_edges).begin()),
          [
            //num_unique_triangles,
            edge_exists = edge_exists.data(),
            edge_q_r = thrust::make_zip_iterator(
            std::get<4>(triangles_from_weak_edges).begin(),
            std::get<5>(triangles_from_weak_edges).begin())
          ] __device__(auto idx) {
            auto src = thrust::get<0>(edge_q_r[idx]);
            auto dst = thrust::get<1>(edge_q_r[idx]);

            return edge_exists[idx] ? thrust::make_tuple(src, dst) : thrust::make_tuple(dst, src);
          }
        );


        auto my_rank = handle.get_comms().get_rank();
        if (my_rank == 1) {
            printf("\nBefore reordering the triangles and after matching DODG edges\n");
            raft::print_device_vector("*triangle_srcs", std::get<0>(triangles_from_weak_edges).data(), std::get<0>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts", std::get<1>(triangles_from_weak_edges).data(), std::get<1>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_srcs_pr", std::get<2>(triangles_from_weak_edges).data(), std::get<2>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts_pr", std::get<3>(triangles_from_weak_edges).data(), std::get<3>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_srcs_qr", std::get<4>(triangles_from_weak_edges).data(), std::get<4>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts_qr", std::get<5>(triangles_from_weak_edges).data(), std::get<5>(triangles_from_weak_edges).size(), std::cout);
        } 


        // re-order triangles
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
                
                } else { // Only for debugging purposes. Remove after.
                  printf("\ninvalid combination\n");
                }

                return triangle;
          }
        );     

        thrust::sort(
          handle.get_thrust_policy(),
          get_dataframe_buffer_begin(triangles_from_weak_edges),
          get_dataframe_buffer_end(triangles_from_weak_edges));
        
        //auto my_rank = handle.get_comms().get_rank();
        if (my_rank == 1) {
            printf("\nAfter re-ordering and sorting triangles\n");
            raft::print_device_vector("*triangle_srcs", std::get<0>(triangles_from_weak_edges).data(), std::get<0>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts", std::get<1>(triangles_from_weak_edges).data(), std::get<1>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_srcs_pr", std::get<2>(triangles_from_weak_edges).data(), std::get<2>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts_pr", std::get<3>(triangles_from_weak_edges).data(), std::get<3>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_srcs_qr", std::get<4>(triangles_from_weak_edges).data(), std::get<4>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts_qr", std::get<5>(triangles_from_weak_edges).data(), std::get<5>(triangles_from_weak_edges).size(), std::cout);
        }

        auto unique_triangle_end =  thrust::unique(
                                        handle.get_thrust_policy(),
                                        get_dataframe_buffer_begin(triangles_from_weak_edges),
                                        get_dataframe_buffer_end(triangles_from_weak_edges));

        auto num_unique_triangles = thrust::distance(get_dataframe_buffer_begin(triangles_from_weak_edges), unique_triangle_end);

        resize_dataframe_buffer(triangles_from_weak_edges, num_unique_triangles, handle.get_stream());

        //auto my_rank = handle.get_comms().get_rank();
        if (my_rank == 1) {
            printf("\nAfter reducing triangles\n");
            raft::print_device_vector("*triangle_srcs", std::get<0>(triangles_from_weak_edges).data(), std::get<0>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts", std::get<1>(triangles_from_weak_edges).data(), std::get<1>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_srcs_pr", std::get<2>(triangles_from_weak_edges).data(), std::get<2>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts_pr", std::get<3>(triangles_from_weak_edges).data(), std::get<3>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_srcs_qr", std::get<4>(triangles_from_weak_edges).data(), std::get<4>(triangles_from_weak_edges).size(), std::cout);
            raft::print_device_vector("*triangle_dsts_qr", std::get<5>(triangles_from_weak_edges).data(), std::get<5>(triangles_from_weak_edges).size(), std::cout);
        }

        // If multi_gpu, shuffle and reduce once more

        if constexpr (multi_gpu) {

          auto& comm                 = handle.get_comms();
          auto const comm_size       = comm.get_size();
          auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
          auto const major_comm_size = major_comm.get_size();
          auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
          auto const minor_comm_size = minor_comm.get_size();

          auto my_rank = handle.get_comms().get_rank();
          
          auto vertex_partition_range_lasts = cur_graph_view.vertex_partition_range_lasts(); // FIXME: Make it global variable

          rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                                  handle.get_stream());
          raft::update_device(d_vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.size(),
                          handle.get_stream()); // FIXME: Create an optional parameter instead that can be populated if running MG
          
          //rmm::device_uvector<> vertex_gpu_ids(multi_gpu ? vertices.size() : 0, stream);
          //rmm::device_uvector<size_t> vertex_pos(multi_gpu ? vertices.size() : 0, stream);

          auto x_df = 
          allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t, vertex_t, vertex_t, vertex_t, vertex_t>>(
            0,
            handle.get_stream());
          
          
          /*
          std::tie(std::ignore, x_df, std::ignore) =
          groupby_gpu_id_and_shuffle_kv_pairs(
              handle.get_comms(),
              //intersection_indices.begin(),
              //intersection_indices.end(),
              
              thrust::make_zip_iterator(
                std::get<0>(triangles_from_weak_edges).begin(),
                std::get<1>(triangles_from_weak_edges).begin()),
              thrust::make_zip_iterator(
                std::get<0>(triangles_from_weak_edges).end(),
                std::get<1>(triangles_from_weak_edges).end()),
              
              get_dataframe_buffer_begin(triangles_from_weak_edges),
              //intersection_indices.begin(),
              //local_counts.begin(),
              cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
                raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                                  d_vertex_partition_range_lasts.size()),
                major_comm_size,
                minor_comm_size},
              handle.get_stream());
          */


            rmm::device_uvector<vertex_t> v1(0, handle.get_stream());
            rmm::device_uvector<vertex_t> v2(0, handle.get_stream());
            rmm::device_uvector<vertex_t> v3(0, handle.get_stream());
            //std::tie(std::ignore, x_df, std::ignore) =
            //std::tie(x_df, std::ignore) =



            std::tie(triangles_from_weak_edges, std::ignore) = 
              groupby_gpu_id_and_shuffle_values(
                  handle.get_comms(),
                  get_dataframe_buffer_begin(triangles_from_weak_edges),
                  get_dataframe_buffer_end(triangles_from_weak_edges),
                  
                 [key_func = 
                 cugraph::detail::compute_gpu_id_from_int_edge_endpoints_t<vertex_t>{
                    raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
                    comm_size,
                    major_comm_size,
                    minor_comm_size}] __device__(auto val) {return key_func(thrust::get<0>(val), thrust::get<1>(val));},
                   
                    handle.get_stream()
                    );

            
            if (my_rank == 1) {
              auto my_rank = handle.get_comms().get_rank();
              printf("\nAfter shufling triangles\n");
              raft::print_device_vector("fin_triangle_srcs", std::get<0>(x_df).data(), std::get<0>(x_df).size(), std::cout);
              raft::print_device_vector("fin_triangle_dsts", std::get<1>(x_df).data(), std::get<1>(x_df).size(), std::cout);
              raft::print_device_vector("fin_triangle_srcs_pr", std::get<2>(x_df).data(), std::get<2>(x_df).size(), std::cout);
              raft::print_device_vector("fin_triangle_dsts_pr", std::get<3>(x_df).data(), std::get<3>(x_df).size(), std::cout);
              raft::print_device_vector("fin_triangle_srcs_qr", std::get<4>(x_df).data(), std::get<4>(x_df).size(), std::cout);
              raft::print_device_vector("fin_triangle_dsts_qr", std::get<5>(x_df).data(), std::get<5>(x_df).size(), std::cout);
            }


            unique_triangle_end =  thrust::unique(
                                        handle.get_thrust_policy(),
                                        get_dataframe_buffer_begin(triangles_from_weak_edges),
                                        get_dataframe_buffer_end(triangles_from_weak_edges));

            num_unique_triangles = thrust::distance(get_dataframe_buffer_begin(triangles_from_weak_edges), unique_triangle_end);

            resize_dataframe_buffer(triangles_from_weak_edges, num_unique_triangles, handle.get_stream());


            if (my_rank == 1) {
              auto my_rank = handle.get_comms().get_rank();
              printf("\nAfter reducing triangles once more\n");
              raft::print_device_vector("fin_triangle_srcs", std::get<0>(x_df).data(), std::get<0>(x_df).size(), std::cout);
              raft::print_device_vector("fin_triangle_dsts", std::get<1>(x_df).data(), std::get<1>(x_df).size(), std::cout);
              raft::print_device_vector("fin_triangle_srcs_pr", std::get<2>(x_df).data(), std::get<2>(x_df).size(), std::cout);
              raft::print_device_vector("fin_triangle_dsts_pr", std::get<3>(x_df).data(), std::get<3>(x_df).size(), std::cout);
              raft::print_device_vector("fin_triangle_srcs_qr", std::get<4>(x_df).data(), std::get<4>(x_df).size(), std::cout);
              raft::print_device_vector("fin_triangle_dsts_qr", std::get<5>(x_df).data(), std::get<5>(x_df).size(), std::cout);
            }

            // Sort and Reduce once more

            /* working
            std::forward_as_tuple(
              std::tie(v1, v2, v3), std::ignore) = 
              groupby_gpu_id_and_shuffle_values(
                  handle.get_comms(),
                  
                  thrust::make_zip_iterator(thrust::make_tuple(std::get<0>(triangles_from_weak_edges).begin(),
                                                               std::get<1>(triangles_from_weak_edges).begin(),
                                                               std::get<3>(triangles_from_weak_edges).begin())),
                  
                  thrust::make_zip_iterator(thrust::make_tuple(std::get<0>(triangles_from_weak_edges).end(),
                                                               std::get<1>(triangles_from_weak_edges).end(),
                                                               std::get<3>(triangles_from_weak_edges).end())),
                 [key_func = 
                 cugraph::detail::compute_gpu_id_from_int_edge_endpoints_t<vertex_t>{
                    raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
                    comm_size,
                    major_comm_size,
                    minor_comm_size}] __device__(auto val) {return key_func(thrust::get<0>(val), thrust::get<1>(val));},
                   
                    handle.get_stream()
                    );
            */
          
          

        }

        auto edgelist_to_update_count =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(3* num_unique_triangles,
                                                                         handle.get_stream());
        

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
        

        // If multi-GPU, shuffle and reduce
        if constexpr (multi_gpu) {

          auto& comm                 = handle.get_comms();
          auto const comm_size       = comm.get_size();
          auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
          auto const major_comm_size = major_comm.get_size();
          auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
          auto const minor_comm_size = minor_comm.get_size();

          auto vertex_partition_range_lasts = cur_graph_view.vertex_partition_range_lasts(); // FIXME: Make it global variable

          rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                                  handle.get_stream());
          raft::update_device(d_vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.size(),
                          handle.get_stream()); // FIXME: Create an optional parameter instead that can be populated if running MG

          auto my_rank = handle.get_comms().get_rank();

          std::tie(edgelist_to_update_count, std::ignore) = 
              groupby_gpu_id_and_shuffle_values(
                  handle.get_comms(),
                  get_dataframe_buffer_begin(edgelist_to_update_count),
                  get_dataframe_buffer_end(edgelist_to_update_count),
                  
                 [key_func = 
                 cugraph::detail::compute_gpu_id_from_int_edge_endpoints_t<vertex_t>{
                    raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
                    comm_size,
                    major_comm_size,
                    minor_comm_size}] __device__(auto val) {return key_func(thrust::get<0>(val), thrust::get<1>(val));},
                   
                    handle.get_stream()
                    );
          
          if (my_rank == 1) {
             printf("\nbefore sorting\n");
            raft::print_device_vector("edgelist_to_update_count_srcs", std::get<0>(edgelist_to_update_count).data(), std::get<0>(edgelist_to_update_count).size(), std::cout);
            raft::print_device_vector("edgelist_to_update_count_dsts", std::get<1>(edgelist_to_update_count).data(), std::get<1>(edgelist_to_update_count).size(), std::cout);
          }
        }

        thrust::sort(
          handle.get_thrust_policy(),
          get_dataframe_buffer_begin(edgelist_to_update_count),
          get_dataframe_buffer_end(edgelist_to_update_count));
        

        if (my_rank == 1) {
            printf("\nafter sorting\n");
          raft::print_device_vector("edgelist_to_update_count_srcs", std::get<0>(edgelist_to_update_count).data(), std::get<0>(edgelist_to_update_count).size(), std::cout);
          raft::print_device_vector("edgelist_to_update_count_dsts", std::get<1>(edgelist_to_update_count).data(), std::get<1>(edgelist_to_update_count).size(), std::cout);
        }
        
        unique_pair_count = thrust::unique_count(handle.get_thrust_policy(),
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

        if (my_rank == 1) {
          printf("\nafter reducing and count\n");
          raft::print_device_vector("vertex_pair_buffer_unique_srcs", std::get<0>(vertex_pair_buffer_unique).data(), std::get<0>(vertex_pair_buffer_unique).size(), std::cout);
          raft::print_device_vector("vertex_pair_buffer_unique_dsts", std::get<1>(vertex_pair_buffer_unique).data(), std::get<1>(vertex_pair_buffer_unique).size(), std::cout);
          raft::print_device_vector("decrease_count", decrease_count.data(), decrease_count.size(), std::cout);
        }
        
        
        // Update count of weak edges

        edges_to_decrement_count.clear();

        edges_to_decrement_count.insert(std::get<0>(vertex_pair_buffer_unique).begin(),
                                        std::get<0>(vertex_pair_buffer_unique).end(),
                                        std::get<1>(vertex_pair_buffer_unique).begin());
        
        cugraph::transform_e(
          handle,
          cur_graph_view,
          edges_to_decrement_count,
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          edge_triangle_counts.view(),
          [
            edge_buffer_first = thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_unique).begin(), std::get<1>(vertex_pair_buffer_unique).begin()),
            edge_buffer_last = thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_unique).end(), std::get<1>(vertex_pair_buffer_unique).end()),
            decrease_count = raft::device_span<edge_t>(decrease_count.data(), decrease_count.size())
          ]
          __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) {
          
            auto itr_pair = thrust::lower_bound(
              thrust::seq, edge_buffer_first, edge_buffer_last, thrust::make_tuple(src, dst));

            auto idx_pair = thrust::distance(edge_buffer_first, itr_pair);

            count -= decrease_count[idx_pair];

          return count;        

          },
          edge_triangle_counts.mutable_view(),
          false); // FIXME: set expensive check to False
    
        edgelist_weak.clear();

        thrust::sort(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin()),
          thrust::make_zip_iterator(weak_edgelist_srcs.end(), weak_edgelist_dsts.end())
        );

        edgelist_weak.insert(weak_edgelist_srcs.begin(),
                             weak_edgelist_srcs.end(),
                             weak_edgelist_dsts.begin());

        // Get undirected graph view

        cur_graph_view.clear_edge_mask();

        cur_graph_view.attach_edge_mask(weak_edges_mask.view());

        auto prev_number_of_edges = cur_graph_view.compute_number_of_edges(handle); // FIXME: rename 'prev_number_of_edges' to 'cur_number_of_edges' ?

        cugraph::transform_e(
            handle,
            cur_graph_view,
            edgelist_weak,
            cugraph::edge_src_dummy_property_t{}.view(),
            cugraph::edge_dst_dummy_property_t{}.view(),
            cugraph::edge_dummy_property_t{}.view(),
            [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {
              
              return false;
            },
            weak_edges_mask.mutable_view(),
            false);
        
        edgelist_weak.clear();
        thrust::sort(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(weak_edgelist_dsts.begin(), weak_edgelist_srcs.begin()),
          thrust::make_zip_iterator(weak_edgelist_dsts.end(), weak_edgelist_srcs.end())
        );
        
        edgelist_weak.insert(weak_edgelist_dsts.begin(),
                             weak_edgelist_dsts.end(),
                             weak_edgelist_srcs.begin());
        
        cugraph::transform_e(
            handle,
            cur_graph_view,
            edgelist_weak,
            cugraph::edge_src_dummy_property_t{}.view(),
            cugraph::edge_dst_dummy_property_t{}.view(),
            cugraph::edge_dummy_property_t{}.view(),
            [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {

              return false;
            },
            weak_edges_mask.mutable_view(),
            false);
        
        cur_graph_view.attach_edge_mask(weak_edges_mask.view());

        if (prev_number_of_edges == cur_graph_view.compute_number_of_edges(handle)) { break; }
      
    }
    

    cur_graph_view.clear_edge_mask();
    cur_graph_view.attach_edge_mask(dodg_mask.view());
   
    cugraph::transform_e(
            handle,
            cur_graph_view,
            cugraph::edge_src_dummy_property_t{}.view(),
            cugraph::edge_dst_dummy_property_t{}.view(),
            edge_triangle_counts.view(),
            //cugraph::edge_dummy_property_t{}.view(),
            [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto count) {
              return count == 0 ? false : true;
            },
            dodg_mask.mutable_view(),
            true);
    
    rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> edgelist_wgts{std::nullopt};

    
    std::tie(edgelist_srcs, edgelist_dsts, edgelist_wgts, std::ignore, std::ignore) =
      decompress_to_edgelist(
        handle,
        cur_graph_view,
        edge_weight_view,
        std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
        renumber_map
          ? std::make_optional(
              raft::device_span<vertex_t const>((*renumber_map).data(), (*renumber_map).size())): // Update renumbering if it exist. 
          std::nullopt
        );
    
    std::tie(edgelist_srcs, edgelist_dsts, edgelist_wgts) =
      symmetrize_edgelist<vertex_t, weight_t, false, multi_gpu>(handle,
                                                                std::move(edgelist_srcs),
                                                                std::move(edgelist_dsts),
                                                                std::move(edgelist_wgts),
                                                                false);

    std::cout << "strong edgelist srcs = " << edgelist_srcs.size() << std::endl;
  
    return std::make_tuple(
      std::move(edgelist_srcs), std::move(edgelist_dsts), std::move(edgelist_wgts));
  }
}
}  // namespace cugraph
