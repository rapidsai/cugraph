/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "prims/extract_transform_v_frontier_outgoing_e.cuh"
#include "prims/edge_bucket.cuh"
#include "prims/extract_transform_e.cuh"
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

template <typename vertex_t, typename edge_t, typename EdgeIterator, bool is_q_r_edge>
edge_t remove_overcompensating_edges(raft::handle_t const& handle,
                                     size_t buffer_size,
                                     EdgeIterator potential_closing_or_incoming_edges,
                                     EdgeIterator incoming_or_potential_closing_edges,
                                     raft::device_span<vertex_t const> weak_edgelist_first_srcs,
                                     raft::device_span<vertex_t const> weak_edgelist_first_dsts)
{
  // To avoid over-compensating, check whether the 'potential_closing_edges'
  // are within the weak edges. If yes, the was already unrolled
  auto edges_not_overcomp = thrust::remove_if(
    handle.get_thrust_policy(),
    thrust::make_zip_iterator(potential_closing_or_incoming_edges,
                              incoming_or_potential_closing_edges),
    thrust::make_zip_iterator(potential_closing_or_incoming_edges + buffer_size,
                              incoming_or_potential_closing_edges + buffer_size),
    [num_weak_edges = weak_edgelist_first_dsts.size(),
     weak_first =
       thrust::make_zip_iterator(weak_edgelist_first_srcs.begin(), weak_edgelist_first_dsts.begin()),
     weak_last = thrust::make_zip_iterator(weak_edgelist_first_srcs.end(),
                                              weak_edgelist_first_dsts.end())] __device__(auto e) {
      auto potential_edge = thrust::get<0>(e);
      auto potential_or_incoming_edge = thrust::make_tuple(thrust::get<0>(potential_edge), thrust::get<1>(potential_edge));
      if constexpr (is_q_r_edge) {
        potential_or_incoming_edge = thrust::make_tuple(thrust::get<1>(potential_edge), thrust::get<0>(potential_edge));
      };

      auto itr = thrust::lower_bound(
        thrust::seq, weak_first, weak_last, potential_or_incoming_edge);
      return (itr != weak_last && *itr == potential_or_incoming_edge);
    });

  auto dist = thrust::distance(thrust::make_zip_iterator(potential_closing_or_incoming_edges,
                                                         incoming_or_potential_closing_edges),
                               edges_not_overcomp);

  return dist;
}

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
struct extract_edges {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t, edge_t>> operator()(
    
    auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto count) const
  {
    return thrust::make_tuple(src, dst, count);
  }
};

template <typename vertex_t>
struct extract_edges_to_q_r {

  raft::device_span<vertex_t const> vertex_q_r{};
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(
  
  auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
  {
    auto itr_src = thrust::find(
        thrust::seq, vertex_q_r.begin(), vertex_q_r.end(), src);

    auto itr_dst = thrust::find(
        thrust::seq, vertex_q_r.begin(), vertex_q_r.end(), dst);

    if (itr_src != vertex_q_r.end() && *itr_src == src) {
      return thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)};
    } else if (itr_dst != vertex_q_r.end() && *itr_dst == dst) {
      return thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)};
    } else {
      return thrust::nullopt;
    }
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

template <typename vertex_t, typename edge_t, bool generate_p_r>
struct generate_p_r_or_q_r_from_p_q {
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

    if constexpr (generate_p_r) {
      return thrust::make_tuple(weak_srcs[chunk_start + idx], intersection_indices[i]);

    } else {
      return thrust::make_tuple(weak_dsts[chunk_start + idx], intersection_indices[i]);
    }
  }
};

template <typename vertex_t, typename edge_t>
struct extract_q_idx {
  using return_type = thrust::optional<thrust::tuple<vertex_t, edge_t>>;

  return_type __device__ operator()(thrust::tuple<vertex_t, edge_t> tagged_src,
                                    vertex_t dst,
                                    thrust::nullopt_t,
                                    thrust::nullopt_t,
                                    thrust::nullopt_t) const
  {
    return thrust::make_optional(thrust::make_tuple(dst, thrust::get<1>(tagged_src)));
  }
};

template <typename vertex_t, typename edge_t>
struct extract_q_idx_closing {
  using return_type = thrust::optional<thrust::tuple<vertex_t, edge_t>>;
  raft::device_span<vertex_t const> weak_edgelist_dsts;

  return_type __device__ operator()(thrust::tuple<vertex_t, edge_t> tagged_src,
                                    vertex_t dst,
                                    thrust::nullopt_t,
                                    thrust::nullopt_t,
                                    thrust::nullopt_t) const
  {
    edge_t idx = thrust::get<1>(tagged_src);
    return dst == weak_edgelist_dsts[idx]
             ? thrust::make_optional(thrust::make_tuple(thrust::get<0>(tagged_src), idx))
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

template <typename vertex_t, typename edge_t, typename EdgeIterator>
struct generate_p_r {
  EdgeIterator weak_edge{};
  raft::device_span<edge_t const> weak_edge_idx{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    return *(weak_edge + weak_edge_idx[i]);
  }
};

template <typename vertex_t, typename edge_t, typename EdgeIterator, bool generate_p_q>
struct generate_p_q_q_r {
  EdgeIterator weak_edge{};
  raft::device_span<vertex_t const> q_closing{};
  raft::device_span<edge_t const> weak_edge_idx{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {

    if constexpr (generate_p_q) {
      return thrust::make_tuple(thrust::get<0>(*(weak_edge + weak_edge_idx[i])), q_closing[i]);
    } else {
      return thrust::make_tuple(q_closing[i], thrust::get<1>(*(weak_edge + weak_edge_idx[i])));
    }
  }
};

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void update_count(raft::handle_t const& handle,
                          graph_view_t<vertex_t, edge_t, false, multi_gpu> & cur_graph_view,
                          edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t> & e_property_triangle_count,
                          raft::device_span<vertex_t> vertex_pair_buffer_src,
                          raft::device_span<vertex_t> vertex_pair_buffer_dst
                          ) {
  
  // Before updating the count, we need to clear the mask
  // cur_graph_view.clear_edge_mask();
  auto vertex_pair_buffer_begin = thrust::make_zip_iterator(vertex_pair_buffer_src.begin(), vertex_pair_buffer_dst.begin());
  
  thrust::sort(handle.get_thrust_policy(),
               vertex_pair_buffer_begin,
               vertex_pair_buffer_begin + vertex_pair_buffer_src.size());
  
  auto unique_pair_count = thrust::unique_count(handle.get_thrust_policy(),
                                                vertex_pair_buffer_begin,
                                                vertex_pair_buffer_begin + vertex_pair_buffer_src.size());
  
  rmm::device_uvector<edge_t> decrease_count(unique_pair_count, handle.get_stream());

  rmm::device_uvector<edge_t> decrease_count_tmp(vertex_pair_buffer_src.size(),
                                                 handle.get_stream());
  
  thrust::fill(handle.get_thrust_policy(),
               decrease_count_tmp.begin(),
               decrease_count_tmp.end(),
               size_t{1});
  
  auto vertex_pair_buffer_unique = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        unique_pair_count, handle.get_stream());
  
  thrust::reduce_by_key(handle.get_thrust_policy(),
                        vertex_pair_buffer_begin,
                        vertex_pair_buffer_begin + vertex_pair_buffer_src.size(),
                        decrease_count_tmp.begin(),
                        get_dataframe_buffer_begin(vertex_pair_buffer_unique),
                        decrease_count.begin(),
                        thrust::equal_to<thrust::tuple<vertex_t, vertex_t>>{});

  cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edges_to_decrement_count(handle);
      edges_to_decrement_count.insert(std::get<0>(vertex_pair_buffer_unique).begin(),
                                  std::get<0>(vertex_pair_buffer_unique).end(),
                                  std::get<1>(vertex_pair_buffer_unique).begin());
  
  cugraph::transform_e(
    handle,
    cur_graph_view,
    edges_to_decrement_count,
    cugraph::edge_src_dummy_property_t{}.view(),
    cugraph::edge_dst_dummy_property_t{}.view(),
    e_property_triangle_count.view(),
    [
      vertex_pair_buffer_begin = get_dataframe_buffer_begin(vertex_pair_buffer_unique),
      vertex_pair_buffer_end = get_dataframe_buffer_end(vertex_pair_buffer_unique),
      decrease_count = decrease_count.data()
    ]
    __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, edge_t count) {
      
      auto e = thrust::make_tuple(src, dst);
      auto itr_pair = thrust::lower_bound(
        thrust::seq, vertex_pair_buffer_begin, vertex_pair_buffer_end, e);

      auto idx_pair = thrust::distance(vertex_pair_buffer_begin, itr_pair);
      return count - decrease_count[idx_pair];
    },
    e_property_triangle_count.mutable_view(),
    true); // FIXME: set expensive check to False 
  
};

template <typename vertex_t, typename edge_t, typename weight_t, typename EdgeIterator, bool multi_gpu>
std::tuple<EdgeIterator, EdgeIterator, EdgeIterator>
accumulate_triangles_p_q_or_q_r(raft::handle_t const& handle,
                              graph_view_t<vertex_t, edge_t, false, multi_gpu> & graph_view,
                              raft::device_span<vertex_t const> weak_edgelist_srcs,
                              raft::device_span<vertex_t const> weak_edgelist_dsts,
                              size_t prev_chunk_size,
                              size_t chunk_size,
                              bool do_expensive_check) {
  
  auto weak_edgelist_first = thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin());

  // Call nbr_intersection unroll (p, q) and (q, r) edges
  auto [intersection_offsets, intersection_indices] =
    detail::nbr_intersection(handle,
                             graph_view,
                             cugraph::edge_dummy_property_t{}.view(),
                             weak_edgelist_first + prev_chunk_size,
                             weak_edgelist_first + prev_chunk_size + chunk_size,
                             std::array<bool, 2>{true, true},
                            //do_expensive_check : FIXME
                            true);

  // Generate (p, q) edges
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
      weak_edgelist_srcs,
      weak_edgelist_dsts
      });
  
  auto vertex_pair_buffer_p_r_edge_p_q =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                      handle.get_stream());
  thrust::tabulate(
    handle.get_thrust_policy(),
    get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_p_q),
    get_dataframe_buffer_end(vertex_pair_buffer_p_r_edge_p_q),
    generate_p_r_or_q_r_from_p_q<vertex_t, edge_t, true>{
      prev_chunk_size,
      raft::device_span<size_t const>(intersection_offsets.data(),
                                      intersection_offsets.size()),
      raft::device_span<vertex_t const>(intersection_indices.data(),
                                        intersection_indices.size()),
      weak_edgelist_srcs,
      weak_edgelist_dsts});
    
  auto vertex_pair_buffer_q_r_edge_p_q =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(intersection_indices.size(),
                                                                      handle.get_stream());
  thrust::tabulate(
    handle.get_thrust_policy(),
    get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q),
    get_dataframe_buffer_end(vertex_pair_buffer_q_r_edge_p_q),
    generate_p_r_or_q_r_from_p_q<vertex_t, edge_t, false>{
      prev_chunk_size,
      raft::device_span<size_t const>(intersection_offsets.data(),
                                      intersection_offsets.size()),
      raft::device_span<vertex_t const>(intersection_indices.data(),
                                        intersection_indices.size()),
      weak_edgelist_srcs,
      weak_edgelist_dsts});

  return std::make_tuple(std::move(vertex_pair_buffer_p_q), std::move(vertex_pair_buffer_p_r_edge_p_q), std::move(vertex_pair_buffer_q_r_edge_p_q));
}

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

  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "Invalid input arguments: K-truss currently supports undirected graphs only.");
  CUGRAPH_EXPECTS(!graph_view.is_multigraph(),
                  "Invalid input arguments: K-truss currently does not support multi-graphs.");

  if (do_expensive_check) {
    // nothing to do
  }

  std::optional<graph_t<vertex_t, edge_t, false, multi_gpu>> modified_graph{std::nullopt};
  std::optional<graph_view_t<vertex_t, edge_t, false, multi_gpu>> modified_graph_view{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, weight_t>>
    edge_weight{std::nullopt};

  if (graph_view.count_self_loops(handle) > edge_t{0}) {
    auto [srcs, dsts] = extract_transform_e(handle,
                                            graph_view,
                                            edge_src_dummy_property_t{}.view(),
                                            edge_dst_dummy_property_t{}.view(),
                                            edge_dummy_property_t{}.view(),
                                            exclude_self_loop_t<vertex_t>{});

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore, std::ignore, std::ignore) =
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

  // 3. Find (k-1)-core and exclude edges that do not belong to (k-1)-core
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
      std::tie(srcs, dsts, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t>(
          handle, std::move(srcs), std::move(dsts), std::nullopt, std::nullopt, std::nullopt);
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
  //#endif

  // 4. Keep only the edges from a low-degree vertex to a high-degree vertex.

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
    update_edge_src_property(handle, cur_graph_view, out_degrees.begin(), edge_src_out_degrees);
    update_edge_dst_property(handle, cur_graph_view, out_degrees.begin(), edge_dst_out_degrees);

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
      std::tie(srcs, dsts, wgts, std::ignore, std::ignore) =
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

  // 5. Decompress the resulting graph to an edges list and ind intersection of edges endpoints
  // for each partition using detail::nbr_intersection

  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;

    edge_weight_view =
      edge_weight ? std::make_optional((*edge_weight).view())
                  : std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt};
    
    auto e_property_triangle_count = edge_triangle_count<vertex_t, edge_t, multi_gpu>(handle, cur_graph_view);

    cugraph::edge_property_t<decltype(cur_graph_view), bool> edge_mask(handle, cur_graph_view);
    cugraph::fill_edge_property(handle, cur_graph_view, true, edge_mask);

    auto iteration = -1;
    while (true) {
      iteration += 1;
      if (iteration == 1) {
        break; // FIXME: Only for debugging purposes
      }
      // extract the edges that have counts less than k - 2. THose edges will be unrolled

      auto [weak_edgelist_srcs, weak_edgelist_dsts] = extract_transform_e(handle,
                                                                          cur_graph_view,
                                                                          edge_src_dummy_property_t{}.view(),
                                                                          edge_dst_dummy_property_t{}.view(),
                                                                          e_property_triangle_count.view(),
                                                                          extract_weak_edges<vertex_t, edge_t>{k});
      auto done = 1;
      if constexpr (multi_gpu)
      {
        if (weak_edgelist_srcs.size() == 0) { done = 0; }
        done = host_scalar_allreduce(
          handle.get_comms(), done, raft::comms::op_t::MAX, handle.get_stream());

        if (done == 0) { break; }
      }
      else if (weak_edgelist_srcs.size() == 0)
      {
        break;
      }
      auto weak_edgelist_first = thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin());
      thrust::sort(handle.get_thrust_policy(),
                          weak_edgelist_first,
                          weak_edgelist_first + weak_edgelist_srcs.size());
      
      // Find intersection edges
      size_t prev_chunk_size         = 0;
      size_t chunk_num_weak_edges = weak_edgelist_srcs.size();
      size_t edges_to_intersect_per_iteration =
            static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 17);
      edges_to_intersect_per_iteration = 100; // FIXME: For testing
      auto num_chunks =
        raft::div_rounding_up_safe(weak_edgelist_srcs.size(), edges_to_intersect_per_iteration);
      
      for (size_t i = 0; i < num_chunks; ++i) {
        auto chunk_size = std::min(edges_to_intersect_per_iteration, chunk_num_weak_edges);
        auto [vertex_pair_buffer_p_q, vertex_pair_buffer_p_r_edge_p_q, vertex_pair_buffer_q_r_edge_p_q] = accumulate_triangles_p_q_or_q_r<vertex_t, edge_t, weight_t, decltype(allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(size_t{0}, handle.get_stream())), multi_gpu>(
                    handle,
                    cur_graph_view,
                    raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
                    raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size()),
                    prev_chunk_size,
                    chunk_size, 
                    do_expensive_check);
        
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_r_edge_p_q_srcs(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_r_edge_p_q_dsts(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_q_r_edge_p_q_srcs(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_q_r_edge_p_q_dsts(0, handle.get_stream());
        // Shuffle edges
        if constexpr (multi_gpu) {
          // FIXME: Check whether we need to shuffle (p, q) edges
          std::tie(vertex_pair_buffer_p_r_edge_p_q_srcs, vertex_pair_buffer_p_r_edge_p_q_dsts, std::ignore, std::ignore, std::ignore) =
          detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                         edge_t,
                                                                                         weight_t,
                                                                                         int32_t>(
            handle,
            std::move(std::get<0>(vertex_pair_buffer_p_r_edge_p_q)),
            std::move(std::get<1>(vertex_pair_buffer_p_r_edge_p_q)),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            cur_graph_view.vertex_partition_range_lasts());

          std::tie(vertex_pair_buffer_q_r_edge_p_q_srcs, vertex_pair_buffer_q_r_edge_p_q_dsts, std::ignore, std::ignore, std::ignore) =
            detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                            edge_t,
                                                                                            weight_t,
                                                                                            int32_t>(
              handle,
              std::move(std::get<0>(vertex_pair_buffer_q_r_edge_p_q)),
              std::move(std::get<1>(vertex_pair_buffer_q_r_edge_p_q)),
              std::nullopt,
              std::nullopt,
              std::nullopt,
              cur_graph_view.vertex_partition_range_lasts());
        }

        update_count<vertex_t, edge_t, weight_t, multi_gpu>(
          handle,
          cur_graph_view,
          e_property_triangle_count,
          raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_q).data(), std::get<0>(vertex_pair_buffer_p_q).size()),
          raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_p_q).data(), std::get<1>(vertex_pair_buffer_p_q).size())
        );

        update_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            e_property_triangle_count,
            multi_gpu ? raft::device_span<vertex_t>(vertex_pair_buffer_p_r_edge_p_q_srcs.data(), vertex_pair_buffer_p_r_edge_p_q_srcs.size()) : raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_r_edge_p_q).data(), std::get<0>(vertex_pair_buffer_p_r_edge_p_q).size()), // FIXME: Make sure multi_gpu is properly handles
            multi_gpu ? raft::device_span<vertex_t>(vertex_pair_buffer_p_r_edge_p_q_dsts.data(), vertex_pair_buffer_p_r_edge_p_q_dsts.size()) : raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_r_edge_p_q).data(), std::get<0>(vertex_pair_buffer_p_r_edge_p_q).size()) // FIXME: Make sure multi_gpu is properly handles
          );  

        update_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            e_property_triangle_count,
            multi_gpu ? raft::device_span<vertex_t>(vertex_pair_buffer_p_r_edge_p_q_srcs.data(), vertex_pair_buffer_p_r_edge_p_q_srcs.size()) : raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_q_r_edge_p_q).data(), std::get<0>(vertex_pair_buffer_q_r_edge_p_q).size()),
            multi_gpu ? raft::device_span<vertex_t>(vertex_pair_buffer_p_r_edge_p_q_dsts.data(), vertex_pair_buffer_p_r_edge_p_q_dsts.size()) : raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_q_r_edge_p_q).data(), std::get<0>(vertex_pair_buffer_q_r_edge_p_q).size())
          );
        
        prev_chunk_size += chunk_size;
        chunk_num_weak_edges -= chunk_size;
      }

      // Iterate over unique vertices that appear as either q or r
      rmm::device_uvector<vertex_t> unique_weak_edgelist_srcs(weak_edgelist_srcs.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> unique_weak_edgelist_dsts(weak_edgelist_dsts.size(), handle.get_stream());
      
      // Get unique srcs and dsts
      thrust::copy(handle.get_thrust_policy(),
                   weak_edgelist_srcs.begin(),
                   weak_edgelist_srcs.end(),
                   unique_weak_edgelist_srcs.begin()
                   );

      thrust::copy(handle.get_thrust_policy(),
                   weak_edgelist_dsts.begin(),
                   weak_edgelist_dsts.end(),
                   unique_weak_edgelist_dsts.begin()
                   );
      
      thrust::sort(handle.get_thrust_policy(), unique_weak_edgelist_srcs.begin(), unique_weak_edgelist_srcs.end());
      
      thrust::sort(handle.get_thrust_policy(), unique_weak_edgelist_dsts.begin(), unique_weak_edgelist_dsts.end());

      auto unique_srcs_end =  thrust::unique(
                                        handle.get_thrust_policy(),
                                        unique_weak_edgelist_srcs.begin(),
                                        unique_weak_edgelist_srcs.end());
      
      auto unique_dsts_end =  thrust::unique(
                                        handle.get_thrust_policy(),
                                        unique_weak_edgelist_dsts.begin(),
                                        unique_weak_edgelist_dsts.end());
      
      auto num_unique_weak_edgelist_srcs = thrust::distance(unique_weak_edgelist_srcs.begin(), unique_srcs_end);
      auto num_unique_weak_edgelist_dsts = thrust::distance(unique_weak_edgelist_dsts.begin(), unique_dsts_end);
      unique_weak_edgelist_srcs.resize(num_unique_weak_edgelist_srcs, handle.get_stream());
      unique_weak_edgelist_dsts.resize(num_unique_weak_edgelist_dsts, handle.get_stream());

      rmm::device_uvector<vertex_t> vertex_q_r(num_unique_weak_edgelist_srcs + num_unique_weak_edgelist_dsts, handle.get_stream());

      thrust::set_union(handle.get_thrust_policy(),
                    unique_weak_edgelist_srcs.begin(),
                    unique_weak_edgelist_srcs.end(),
                    unique_weak_edgelist_dsts.begin(),
                    unique_weak_edgelist_dsts.end(),
                    vertex_q_r.begin());

      thrust::sort(handle.get_thrust_policy(), vertex_q_r.begin(), vertex_q_r.end());

      auto weak_unique_v_end =  thrust::unique(
                                      handle.get_thrust_policy(),
                                      vertex_q_r.begin(),
                                      vertex_q_r.end());
      
      vertex_q_r.resize(thrust::distance(vertex_q_r.begin(), weak_unique_v_end), handle.get_stream());

      weak_edgelist_first = thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin()); // FIXME: is this necessary ?

      auto [srcs_to_q_r, dsts_to_q_r] = extract_transform_e(handle,
                                                cur_graph_view,
                                                cugraph::edge_src_dummy_property_t{}.view(),
                                                cugraph::edge_dst_dummy_property_t{}.view(),
                                                cugraph::edge_dummy_property_t{}.view(),
                                                extract_edges_to_q_r<vertex_t>{raft::device_span<vertex_t const>(vertex_q_r.data(), vertex_q_r.size())});
      
      if constexpr (multi_gpu) {
        std::tie(dsts_to_q_r, srcs_to_q_r, std::ignore, std::ignore, std::ignore) =
          detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                         edge_t,
                                                                                         weight_t,
                                                                                         int32_t>(
            handle,
            std::move(dsts_to_q_r),
            std::move(srcs_to_q_r),
            std::nullopt,
            std::nullopt,
            std::nullopt);
      }
      
      std::optional<graph_t<vertex_t, edge_t, false, multi_gpu>> graph_q_r{std::nullopt};
      std::optional<rmm::device_uvector<vertex_t>> renumber_map_q_r{std::nullopt};
      std::tie(*graph_q_r, std::ignore, std::ignore, std::ignore, renumber_map_q_r) =
        create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, false, multi_gpu>(
          handle,
          std::nullopt,
          std::move(dsts_to_q_r),
          std::move(srcs_to_q_r),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          cugraph::graph_properties_t{true, graph_view.is_multigraph()},
          true);

      auto csc_q_r_graph_view = (*graph_q_r).view();

      rmm::device_uvector<vertex_t> cp_weak_edgelist_srcs(0, handle.get_stream());
      rmm::device_uvector<vertex_t> cp_weak_edgelist_dsts(0, handle.get_stream());
      size_t weak_edgelist_size = weak_edgelist_srcs.size();

      if constexpr (multi_gpu) {
        cp_weak_edgelist_srcs.resize(weak_edgelist_srcs.size(), handle.get_stream());
        cp_weak_edgelist_dsts.resize(weak_edgelist_dsts.size(), handle.get_stream());

        thrust::copy(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin()),
          thrust::make_zip_iterator(weak_edgelist_srcs.end(), weak_edgelist_dsts.end()),
          thrust::make_zip_iterator(cp_weak_edgelist_srcs.begin(), cp_weak_edgelist_dsts.begin()));

        rmm::device_uvector<vertex_t> shuffled_weak_edgelist_srcs{0, handle.get_stream()};
        rmm::device_uvector<vertex_t> shuffled_weak_edgelist_dsts{0, handle.get_stream()};

        std::tie(
          cp_weak_edgelist_srcs, cp_weak_edgelist_dsts, std::ignore, std::ignore, std::ignore) =
          detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                         edge_t,
                                                                                         weight_t,
                                                                                         int32_t>(
            handle,
            std::move(cp_weak_edgelist_srcs),
            std::move(cp_weak_edgelist_dsts),
            std::nullopt,
            std::nullopt,
            std::nullopt);

        renumber_ext_vertices<vertex_t, multi_gpu>(
          handle,
          cp_weak_edgelist_srcs.data(),
          cp_weak_edgelist_srcs.size(),
          (*renumber_map_q_r).data(),
          csc_q_r_graph_view.local_vertex_partition_range_first(),
          csc_q_r_graph_view.local_vertex_partition_range_last(),
          true);

        renumber_ext_vertices<vertex_t, multi_gpu>(
          handle,
          cp_weak_edgelist_dsts.data(),
          cp_weak_edgelist_dsts.size(),
          (*renumber_map_q_r).data(),
          csc_q_r_graph_view.local_vertex_partition_range_first(),
          csc_q_r_graph_view.local_vertex_partition_range_last(),
          true);
        
        weak_edgelist_size = cp_weak_edgelist_srcs.size();
        weak_edgelist_first =
        thrust::make_zip_iterator(cp_weak_edgelist_srcs.begin(), cp_weak_edgelist_dsts.begin());
        thrust::sort(handle.get_thrust_policy(),
                    weak_edgelist_first,
                    weak_edgelist_first + cp_weak_edgelist_srcs.size());
      }
      prev_chunk_size         = 0;
      chunk_num_weak_edges = weak_edgelist_size;

      num_chunks =
        raft::div_rounding_up_safe(weak_edgelist_size, edges_to_intersect_per_iteration);
      
      auto sorted_weak_edgelist_srcs = thrust::get<0>(weak_edgelist_first.get_iterator_tuple());
      auto sorted_weak_edgelist_dsts = thrust::get<1>(weak_edgelist_first.get_iterator_tuple());

      for (size_t i = 0; i < num_chunks; ++i) {
        auto chunk_size = std::min(edges_to_intersect_per_iteration, chunk_num_weak_edges);
  
        // Find intersection of weak edges
        auto [vertex_pair_buffer_q_r, vertex_pair_buffer_p_q_edge_q_r, vertex_pair_buffer_p_r_edge_q_r] = accumulate_triangles_p_q_or_q_r<vertex_t, edge_t, weight_t, decltype(allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(size_t{0}, handle.get_stream())), multi_gpu>(
                    handle,
                    csc_q_r_graph_view,
                    raft::device_span<vertex_t const>(sorted_weak_edgelist_srcs, weak_edgelist_size),
                    raft::device_span<vertex_t const>(sorted_weak_edgelist_dsts, weak_edgelist_size),
                    prev_chunk_size,
                    chunk_size,
                    do_expensive_check);

        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_q_edge_q_r_srcs(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_q_edge_q_r_dsts(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_r_edge_q_r_srcs(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_r_edge_q_r_dsts(0, handle.get_stream());
        if constexpr (multi_gpu) {

          // Unrenumber
          auto vertex_partition_range_lasts = std::make_optional<std::vector<vertex_t>>(
          csc_q_r_graph_view.vertex_partition_range_lasts());

          unrenumber_int_vertices<vertex_t, multi_gpu>(
            handle,
            std::get<0>(vertex_pair_buffer_p_q_edge_q_r).data(),
            std::get<0>(vertex_pair_buffer_p_q_edge_q_r).size(),
            (*renumber_map_q_r).data(),
            *vertex_partition_range_lasts,
            true);

          unrenumber_int_vertices<vertex_t, multi_gpu>(
            handle,
            std::get<1>(vertex_pair_buffer_p_q_edge_q_r).data(),
            std::get<1>(vertex_pair_buffer_p_q_edge_q_r).size(),
            (*renumber_map_q_r).data(),
            *vertex_partition_range_lasts,
            true);

          unrenumber_int_vertices<vertex_t, multi_gpu>(
            handle,
            std::get<0>(vertex_pair_buffer_p_r_edge_q_r).data(),
            std::get<0>(vertex_pair_buffer_p_r_edge_q_r).size(),
            (*renumber_map_q_r).data(),
            *vertex_partition_range_lasts,
            true);

          unrenumber_int_vertices<vertex_t, multi_gpu>(
            handle,
            std::get<1>(vertex_pair_buffer_p_r_edge_q_r).data(),
            std::get<1>(vertex_pair_buffer_p_r_edge_q_r).size(),
            (*renumber_map_q_r).data(),
            *vertex_partition_range_lasts,
            true);

          unrenumber_int_vertices<vertex_t, multi_gpu>(handle,
                                                      std::get<0>(vertex_pair_buffer_q_r).data(),
                                                      std::get<0>(vertex_pair_buffer_q_r).size(),
                                                      (*renumber_map_q_r).data(),
                                                      *vertex_partition_range_lasts,
                                                      true);

          unrenumber_int_vertices<vertex_t, multi_gpu>(handle,
                                                      std::get<1>(vertex_pair_buffer_q_r).data(),
                                                      std::get<1>(vertex_pair_buffer_q_r).size(),
                                                      (*renumber_map_q_r).data(),
                                                      *vertex_partition_range_lasts,
                                                      true);

          auto& comm           = handle.get_comms();
          auto const comm_rank = comm.get_rank();  // FIXME: for debugging
          // Get global weak_edgelist
          // FIXME: Perform all-to-all in chunks
          auto global_weak_edgelist_srcs = cugraph::detail::device_allgatherv(
            handle, comm, raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()));
          // FIXME: Perform all-to-all in chunks
          auto global_weak_edgelist_dsts = cugraph::detail::device_allgatherv(
            handle, comm, raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size()));
        
          weak_edgelist_size = global_weak_edgelist_srcs.size();
          // Sort the weak edges if they are not already
          weak_edgelist_first =
            thrust::make_zip_iterator(global_weak_edgelist_srcs.begin(), global_weak_edgelist_dsts.begin());
          thrust::sort(handle.get_thrust_policy(),
                       weak_edgelist_first,
                       weak_edgelist_first + weak_edgelist_srcs.size());
        }


        sorted_weak_edgelist_srcs = thrust::get<0>(weak_edgelist_first.get_iterator_tuple());
        sorted_weak_edgelist_dsts = thrust::get<1>(weak_edgelist_first.get_iterator_tuple());
        auto num_edges_not_overcomp =
            remove_overcompensating_edges<vertex_t,
                                          edge_t,
                                          decltype(get_dataframe_buffer_begin(vertex_pair_buffer_q_r)),
                                          true>(
              handle,
              size_dataframe_buffer(vertex_pair_buffer_p_q_edge_q_r),
              get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_q_r),
              get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_q_r),
              raft::device_span<vertex_t const>(sorted_weak_edgelist_srcs, weak_edgelist_size),
              raft::device_span<vertex_t const>(sorted_weak_edgelist_dsts, weak_edgelist_size)
              );
        
        resize_dataframe_buffer(vertex_pair_buffer_p_q_edge_q_r, num_edges_not_overcomp, handle.get_stream());
        resize_dataframe_buffer(vertex_pair_buffer_p_r_edge_q_r, num_edges_not_overcomp, handle.get_stream());

        // resize initial (q, r) edges
        resize_dataframe_buffer(vertex_pair_buffer_q_r, num_edges_not_overcomp, handle.get_stream());
        // Reconstruct (q, r) edges that didn't already have their count updated
        thrust::tabulate(
          handle.get_thrust_policy(),
          get_dataframe_buffer_begin(vertex_pair_buffer_q_r), // FIXME: Properly reconstruct (p, r) even when there is no overcompensation ************************************
          get_dataframe_buffer_end(vertex_pair_buffer_q_r),
          [
            vertex_pair_buffer_p_q_edge_q_r = get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_q_r),
            vertex_pair_buffer_p_r_edge_q_r = get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_q_r)
          ] __device__(auto i) {
            return thrust::make_tuple(thrust::get<0>(vertex_pair_buffer_p_q_edge_q_r[i]), thrust::get<0>(vertex_pair_buffer_p_r_edge_q_r[i]));
          });

        if constexpr (multi_gpu) {        
          // Shuffle before updating count
          rmm::device_uvector<vertex_t> vertex_pair_buffer_q_r_srcs(0, handle.get_stream());
          rmm::device_uvector<vertex_t> vertex_pair_buffer_q_r_dsts(0, handle.get_stream());
        
          std::tie(vertex_pair_buffer_q_r_srcs, vertex_pair_buffer_q_r_dsts, std::ignore, std::ignore, std::ignore) =
          detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                        edge_t,
                                                                                        weight_t,
                                                                                        int32_t>(
            handle,
            std::move(std::get<0>(vertex_pair_buffer_q_r)),
            std::move(std::get<1>(vertex_pair_buffer_q_r)),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            cur_graph_view.vertex_partition_range_lasts());
      
          update_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            e_property_triangle_count,
            raft::device_span<vertex_t>(vertex_pair_buffer_q_r_srcs.data(), vertex_pair_buffer_q_r_srcs.size()),
            raft::device_span<vertex_t>(vertex_pair_buffer_q_r_dsts.data(), vertex_pair_buffer_q_r_dsts.size())
          );

          // Shuffle before updating count
          rmm::device_uvector<vertex_t> vertex_pair_buffer_p_q_edge_q_r_srcs(0, handle.get_stream());
          rmm::device_uvector<vertex_t> vertex_pair_buffer_p_q_edge_q_r_dsts(0, handle.get_stream());
          
          std::tie(vertex_pair_buffer_p_q_edge_q_r_dsts, vertex_pair_buffer_p_q_edge_q_r_srcs, std::ignore, std::ignore, std::ignore) =
          detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                        edge_t,
                                                                                        weight_t,
                                                                                        int32_t>(
            handle,
            std::move(std::get<1>(vertex_pair_buffer_p_q_edge_q_r)),
            std::move(std::get<0>(vertex_pair_buffer_p_q_edge_q_r)),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            cur_graph_view.vertex_partition_range_lasts());

          update_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            e_property_triangle_count,
            raft::device_span<vertex_t>(vertex_pair_buffer_p_q_edge_q_r_dsts.data(), vertex_pair_buffer_p_q_edge_q_r_dsts.size()),
            raft::device_span<vertex_t>(vertex_pair_buffer_p_q_edge_q_r_srcs.data(), vertex_pair_buffer_p_q_edge_q_r_srcs.size())
          );

          // Shuffle before updating count
          rmm::device_uvector<vertex_t> vertex_pair_buffer_p_r_edge_q_r_srcs(0, handle.get_stream());
          rmm::device_uvector<vertex_t> vertex_pair_buffer_p_r_edge_q_r_dsts(0, handle.get_stream());
          
          std::tie(vertex_pair_buffer_p_r_edge_q_r_dsts, vertex_pair_buffer_p_r_edge_q_r_srcs, std::ignore, std::ignore, std::ignore) =
          detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                        edge_t,
                                                                                        weight_t,
                                                                                        int32_t>(
            handle,
            std::move(std::get<1>(vertex_pair_buffer_p_r_edge_q_r)),
            std::move(std::get<0>(vertex_pair_buffer_p_r_edge_q_r)),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            cur_graph_view.vertex_partition_range_lasts());
    
          update_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            e_property_triangle_count,
            raft::device_span<vertex_t>(vertex_pair_buffer_p_r_edge_q_r_dsts.data(), vertex_pair_buffer_p_r_edge_q_r_dsts.size()),
            raft::device_span<vertex_t>(vertex_pair_buffer_p_r_edge_q_r_srcs.data(), vertex_pair_buffer_p_r_edge_q_r_srcs.size())
          );
        
        } else {
          update_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            e_property_triangle_count,
            raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_q_r).data(), std::get<0>(vertex_pair_buffer_q_r).size()),
            raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_q_r).data(), std::get<1>(vertex_pair_buffer_q_r).size())
          );
          update_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            e_property_triangle_count,
            raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_p_q_edge_q_r).data(), std::get<0>(vertex_pair_buffer_p_q_edge_q_r).size()),
            raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_q_edge_q_r).data(), std::get<1>(vertex_pair_buffer_p_q_edge_q_r).size())
          );
          update_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            e_property_triangle_count,
            raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_p_r_edge_q_r).data(), std::get<0>(vertex_pair_buffer_p_r_edge_q_r).size()),
            raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_r_edge_q_r).data(), std::get<1>(vertex_pair_buffer_p_r_edge_q_r).size())
          );
        }
        
        prev_chunk_size += chunk_size;
        chunk_num_weak_edges -= chunk_size;
      
      }

      weak_edgelist_first =
            thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin());
      //#if 0
      // Unrolling p, r edges
      // create pair weak_src, weak_edge_idx
      // create a dataframe buffer of size weak_edge_size
      // FIXME: No need to create a dataframe buffer. We can just zip weak_edgelist_srcs
      // with a vector counting from 0 .. 
      auto vertex_pair_buffer_p_tag =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, edge_t>>(weak_edgelist_srcs.size(),
                                                                    handle.get_stream());
      if constexpr (multi_gpu) {
        std::vector<vertex_t> h_num_weak_edges = {vertex_t{weak_edgelist_srcs.size()}};
        rmm::device_uvector<vertex_t> num_weak_edges(1, handle.get_stream());

        raft::update_device(num_weak_edges.data(), h_num_weak_edges.data(), h_num_weak_edges.size(), handle.get_stream());
        
        auto& comm = handle.get_comms();
        auto comm_rank = comm.get_rank();
          // Get global weak_edgelist
        auto global_num_weak_edges = cugraph::detail::device_allgatherv(
          handle,
          comm,
          raft::device_span<vertex_t const>(num_weak_edges.data(), num_weak_edges.size()));
        
        rmm::device_uvector<vertex_t> prefix_sum_global_num_weak_edges(global_num_weak_edges.size(), handle.get_stream());
        thrust::inclusive_scan(handle.get_thrust_policy(),
                         global_num_weak_edges.begin(),
                         global_num_weak_edges.end(),
                         prefix_sum_global_num_weak_edges.begin());
        
        thrust::tabulate(handle.get_thrust_policy(),
                       get_dataframe_buffer_begin(vertex_pair_buffer_p_tag),
                       get_dataframe_buffer_end(vertex_pair_buffer_p_tag),
                       [rank = comm_rank,
                        num_weak_edges = prefix_sum_global_num_weak_edges.begin(),
                        p = weak_edgelist_srcs.begin()] __device__(auto idx) {
                         if (rank != 0) {
                          auto idx_tag = idx + (num_weak_edges[rank - 1]);
                          return thrust::make_tuple(p[idx], idx_tag);
                         }
                         
                         return thrust::make_tuple(p[idx], idx);
                       });
      
      } else {
        thrust::tabulate(
              handle.get_thrust_policy(),
              get_dataframe_buffer_begin(vertex_pair_buffer_p_tag),
              get_dataframe_buffer_end(vertex_pair_buffer_p_tag),
              [
                p = weak_edgelist_srcs.begin()
              ] __device__(auto idx) {
                return thrust::make_tuple(p[idx], idx);
                });
      }
      
      vertex_frontier_t<vertex_t, edge_t, multi_gpu, false> vertex_frontier(handle, 1);
      vertex_frontier.bucket(0).insert(
      thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_p_tag).begin(), std::get<1>(vertex_pair_buffer_p_tag).begin()),
      thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_p_tag).end(), std::get<1>(vertex_pair_buffer_p_tag).end()));

      auto [q, idx] =
        cugraph::extract_transform_v_frontier_outgoing_e(
          handle,
          cur_graph_view,
          vertex_frontier.bucket(0),
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          cugraph::edge_dummy_property_t{}.view(),
          extract_q_idx<vertex_t, edge_t>{},
          do_expensive_check);
      
      vertex_frontier.bucket(0).clear();

      vertex_frontier.bucket(0).insert(
      thrust::make_zip_iterator(q.begin(), idx.begin()),
      thrust::make_zip_iterator(q.end(), idx.end()));

      // FIXME: Need to mask (p, q) and (q, r) edges before unrolling (p, r) edges to avoid overcompensating
      auto [q_closing, idx_closing] =
        cugraph::extract_transform_v_frontier_outgoing_e(
          handle,
          cur_graph_view,
          vertex_frontier.bucket(0),
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          cugraph::edge_dummy_property_t{}.view(),
          extract_q_idx_closing<vertex_t, edge_t>{raft::device_span<vertex_t>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size())},
          do_expensive_check);
      
      // extract pair (p, r)
      auto vertex_pair_buffer_p_r =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(q_closing.size(),
                                                                    handle.get_stream());
      thrust::tabulate(
          handle.get_thrust_policy(),
          get_dataframe_buffer_begin(vertex_pair_buffer_p_r),
          get_dataframe_buffer_end(vertex_pair_buffer_p_r),
          generate_p_r<vertex_t, edge_t, decltype(weak_edgelist_first)>{
            weak_edgelist_first,
            raft::device_span<edge_t const>(idx_closing.data(),
                                            idx_closing.size())
            });
      
      // construct pair (p, q)
      auto vertex_pair_buffer_p_q_for_p_r =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(q_closing.size(),
                                                                    handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_p_q_for_p_r),
        get_dataframe_buffer_end(vertex_pair_buffer_p_q_for_p_r),
        generate_p_q_q_r<vertex_t, edge_t, decltype(weak_edgelist_first), true>{
          weak_edgelist_first,
          raft::device_span<vertex_t const>(q_closing.data(),
                                          q_closing.size()),
          raft::device_span<edge_t const>(idx_closing.data(),
                                          idx_closing.size())
          });

      // construct pair (q, r)
      auto vertex_pair_buffer_q_r_for_p_r =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(q_closing.size(),
                                                                    handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_q_r_for_p_r),
        get_dataframe_buffer_end(vertex_pair_buffer_q_r_for_p_r),
        generate_p_q_q_r<vertex_t, edge_t, decltype(weak_edgelist_first), false>{
          weak_edgelist_first,
          raft::device_span<vertex_t const>(q_closing.data(),
                                          q_closing.size()),
          raft::device_span<edge_t const>(idx_closing.data(),
                                          idx_closing.size())
          });

      auto num_edges_not_overcomp_p_q =
        remove_overcompensating_edges<vertex_t,
                                      edge_t,
                                      decltype(get_dataframe_buffer_begin(vertex_pair_buffer_p_q_for_p_r)),
                                      false>(
          handle,
          q_closing.size(),
          get_dataframe_buffer_begin(vertex_pair_buffer_p_q_for_p_r),
          get_dataframe_buffer_begin(vertex_pair_buffer_q_r_for_p_r),
          raft::device_span<vertex_t const>(sorted_weak_edgelist_srcs, weak_edgelist_size),
          raft::device_span<vertex_t const>(sorted_weak_edgelist_dsts, weak_edgelist_size));

      resize_dataframe_buffer(vertex_pair_buffer_p_q_for_p_r, num_edges_not_overcomp_p_q, handle.get_stream());
      resize_dataframe_buffer(vertex_pair_buffer_q_r_for_p_r, num_edges_not_overcomp_p_q, handle.get_stream());

      auto num_edges_not_overcomp_q_r =
        remove_overcompensating_edges<vertex_t,
                                      edge_t,
                                      decltype(get_dataframe_buffer_begin(vertex_pair_buffer_p_q_for_p_r)),
                                      false>(
          handle,
          num_edges_not_overcomp_p_q,
          get_dataframe_buffer_begin(vertex_pair_buffer_q_r_for_p_r),
          get_dataframe_buffer_begin(vertex_pair_buffer_p_q_for_p_r),
          raft::device_span<vertex_t const>(sorted_weak_edgelist_srcs, weak_edgelist_size),
          raft::device_span<vertex_t const>(sorted_weak_edgelist_dsts, weak_edgelist_size));
      
      resize_dataframe_buffer(vertex_pair_buffer_p_q_for_p_r, num_edges_not_overcomp_q_r, handle.get_stream());
      resize_dataframe_buffer(vertex_pair_buffer_q_r_for_p_r, num_edges_not_overcomp_q_r, handle.get_stream());

      // Reconstruct (p, r) edges that didn't already have their count updated
      resize_dataframe_buffer(vertex_pair_buffer_p_r, num_edges_not_overcomp_q_r, handle.get_stream());
        thrust::tabulate(
          handle.get_thrust_policy(),
          get_dataframe_buffer_begin(vertex_pair_buffer_p_r),
          get_dataframe_buffer_end(vertex_pair_buffer_p_r),
          [
            vertex_pair_buffer_p_q_for_p_r = get_dataframe_buffer_begin(vertex_pair_buffer_p_q_for_p_r),
            vertex_pair_buffer_q_r_for_p_r = get_dataframe_buffer_begin(vertex_pair_buffer_q_r_for_p_r)
          ] __device__(auto i) {
            return thrust::make_tuple(thrust::get<0>(vertex_pair_buffer_p_q_for_p_r[i]), thrust::get<1>(vertex_pair_buffer_q_r_for_p_r[i]));
          });
    
      
      if constexpr (multi_gpu) { 
        // Shuffle before updating count
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_r_srcs(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_r_dsts(0, handle.get_stream());

        std::tie(vertex_pair_buffer_p_r_srcs, vertex_pair_buffer_p_r_dsts, std::ignore, std::ignore, std::ignore) =
          detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                        edge_t,
                                                                                        weight_t,
                                                                                        int32_t>(
            handle,
            std::move(std::get<0>(vertex_pair_buffer_p_r)),
            std::move(std::get<1>(vertex_pair_buffer_p_r)),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            cur_graph_view.vertex_partition_range_lasts());

        update_count<vertex_t, edge_t, weight_t, multi_gpu>(
          handle,
          cur_graph_view,
          e_property_triangle_count,
          raft::device_span<vertex_t>(vertex_pair_buffer_p_r_srcs.data(), vertex_pair_buffer_p_r_srcs.size()),
          raft::device_span<vertex_t>(vertex_pair_buffer_p_r_dsts.data(), vertex_pair_buffer_p_r_dsts.size())
        );

        // Shuffle before updating count
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_q_edge_p_r_srcs(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_q_edge_p_r_dsts(0, handle.get_stream());
        
        std::tie(vertex_pair_buffer_p_q_edge_p_r_dsts, vertex_pair_buffer_p_q_edge_p_r_srcs, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                      edge_t,
                                                                                      weight_t,
                                                                                      int32_t>(
          handle,
          std::move(std::get<0>(vertex_pair_buffer_p_q_for_p_r)), // FIXME: rename to vertex_pair_buffer_p_q_edge_p_r for consistency
          std::move(std::get<1>(vertex_pair_buffer_p_q_for_p_r)), // FIXME: rename to vertex_pair_buffer_p_q_edge_p_r for consistency
          std::nullopt,
          std::nullopt,
          std::nullopt,
          cur_graph_view.vertex_partition_range_lasts());

        update_count<vertex_t, edge_t, weight_t, multi_gpu>(
          handle,
          cur_graph_view,
          e_property_triangle_count,
          raft::device_span<vertex_t>(vertex_pair_buffer_p_q_edge_p_r_srcs.data(), vertex_pair_buffer_p_q_edge_p_r_srcs.size()),
          raft::device_span<vertex_t>(vertex_pair_buffer_p_q_edge_p_r_dsts.data(), vertex_pair_buffer_p_q_edge_p_r_dsts.size())
        );

        // Shuffle before updating count
        rmm::device_uvector<vertex_t> vertex_pair_buffer_q_r_edge_p_r_srcs(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_q_r_edge_p_r_dsts(0, handle.get_stream());
        
        std::tie(vertex_pair_buffer_q_r_edge_p_r_dsts, vertex_pair_buffer_q_r_edge_p_r_srcs, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                      edge_t,
                                                                                      weight_t,
                                                                                      int32_t>(
          handle,
          std::move(std::get<0>(vertex_pair_buffer_q_r_for_p_r)), // FIXME: rename to vertex_pair_buffer_p_q_edge_p_r for consistency
          std::move(std::get<1>(vertex_pair_buffer_q_r_for_p_r)), // FIXME: rename to vertex_pair_buffer_p_q_edge_p_r for consistency
          std::nullopt,
          std::nullopt,
          std::nullopt,
          cur_graph_view.vertex_partition_range_lasts());

        update_count<vertex_t, edge_t, weight_t, multi_gpu>(
          handle,
          cur_graph_view,
          e_property_triangle_count,
          raft::device_span<vertex_t>(vertex_pair_buffer_q_r_edge_p_r_srcs.data(), vertex_pair_buffer_q_r_edge_p_r_srcs.size()),
          raft::device_span<vertex_t>(vertex_pair_buffer_q_r_edge_p_r_dsts.data(), vertex_pair_buffer_q_r_edge_p_r_dsts.size())
        );
      
      } else {
        update_count<vertex_t, edge_t, weight_t, multi_gpu>(
          handle,
          cur_graph_view,
          e_property_triangle_count,
          raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_r).data(), std::get<0>(vertex_pair_buffer_p_r).size()),
          raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_p_r).data(), std::get<1>(vertex_pair_buffer_p_r).size())
        );
        
        update_count<vertex_t, edge_t, weight_t, multi_gpu>(
          handle,
          cur_graph_view,
          e_property_triangle_count,
          raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_q_for_p_r).data(), std::get<0>(vertex_pair_buffer_p_q_for_p_r).size()),
          raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_p_q_for_p_r).data(), std::get<1>(vertex_pair_buffer_p_q_for_p_r).size())
        );
        
        update_count<vertex_t, edge_t, weight_t, multi_gpu>(
          handle,
          cur_graph_view,
          e_property_triangle_count,
          raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_q_r_for_p_r).data(), std::get<0>(vertex_pair_buffer_q_r_for_p_r).size()),
          raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_q_r_for_p_r).data(), std::get<1>(vertex_pair_buffer_q_r_for_p_r).size())
        );
      }

      // Mask all the edges that have 0 count
      cugraph::transform_e(
          handle,
          cur_graph_view,
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          e_property_triangle_count.view(),
          [] __device__(
            auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto count) {
            return count != 0;
          },
          edge_mask.mutable_view(),
          false);

      cur_graph_view.attach_edge_mask(edge_mask.view());

      auto [edgelist_srcs_, edgelist_dsts_, edgelist_wgts_, dummy_0, dummy_1] =
      decompress_to_edgelist(
        handle,
        cur_graph_view,
        edge_weight_view ? std::make_optional(*edge_weight_view) : std::nullopt,
        std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
        std::optional<raft::device_span<vertex_t const>>(std::nullopt));
    }

    rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> edgelist_wgts{std::nullopt};
    #if 0
    std::tie(edgelist_srcs, edgelist_dsts, edgelist_wgts, std::ignore, std::ignore) =
      decompress_to_edgelist(
        handle,
        cur_graph_view,
        edge_weight_view,
        std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
        std::optional<raft::device_span<vertex_t const>>(std::nullopt));
  
    std::tie(edgelist_srcs, edgelist_dsts, edgelist_wgts) =
      symmetrize_edgelist<vertex_t, weight_t, false, multi_gpu>(handle,
                                                                std::move(edgelist_srcs),
                                                                std::move(edgelist_dsts),
                                                                std::move(edgelist_wgts),
                                                                false);

    #endif
    return std::make_tuple(
      std::move(edgelist_srcs), std::move(edgelist_dsts), std::move(edgelist_wgts));
  }

}
}  // namespace cugraph
