/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

// FIXME: remove all unused imports
#include <prims/edge_bucket.cuh>
#include <prims/extract_transform_e.cuh>
#include <prims/reduce_op.cuh>
#include <prims/transform_e.cuh>
#include <prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/optional.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/adjacent_difference.h>

namespace cugraph {

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

template <typename edge_t>
struct in_k_plus_one_or_greater_t {
  edge_t k{};
  __device__ bool operator()(edge_t core_number) const { return core_number >= k-1; }
};

template <typename vertex_t>
struct extract_k_plus_one_core_t {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(
    vertex_t src,
    vertex_t dst,
    bool src_in_k_plus_one_core,
    bool dst_in_k_plus_one_core,
    thrust::nullopt_t) const
  {
    return (src_in_k_plus_one_core && dst_in_k_plus_one_core)
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
             : thrust::nullopt;
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


template <typename vertex_t, typename edge_t, typename EdgeIterator>
struct unroll_edge {
  raft::device_span<vertex_t> num_triangles{};
  EdgeIterator edge_unrolled{};
  EdgeIterator edge_first{};
  EdgeIterator edge_last{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    // edges are sort with destination as key so reverse the edge when looking it
    auto pair =
      thrust::make_tuple(thrust::get<1>(*(edge_unrolled + i)), thrust::get<0>(*(edge_unrolled + i)));        
    // Find its position in 'edges'
    auto itr = thrust::lower_bound(thrust::seq,
                                   edge_first,
                                   edge_last,
                                   pair
                                   );

    auto idx = thrust::distance(edge_first, itr);
    cuda::atomic_ref<vertex_t, cuda::thread_scope_device> atomic_counter(num_triangles[idx]);
    auto r = atomic_counter.fetch_sub(vertex_t{1}, cuda::std::memory_order_relaxed);
    
  }
};


template <typename vertex_t, typename edge_t, typename EdgeIterator>
struct generate_p_r {
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};

  EdgeIterator edge_first{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);
    auto pair =
      thrust::make_tuple(thrust::get<0>(*(edge_first + idx)), intersection_indices[i]);

    return pair;
  }
};


template <typename vertex_t, typename edge_t, typename EdgeIterator>
struct generate_q_r {
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};

  EdgeIterator edge_first{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);
    auto pair =
      thrust::make_tuple(thrust::get<1>(*(edge_first + idx)), intersection_indices[i]);

    return pair;
  }
};

template <typename vertex_t, typename edge_t>
struct intersection_op_t {
  __device__ thrust::tuple<edge_t, edge_t> operator()(
    vertex_t v0,
    vertex_t v1,
    edge_t v0_prop /* out degree */,
    edge_t v1_prop /* out degree */,
    raft::device_span<vertex_t const> intersection,
    std::byte, /* dummy */
    std::byte  /* dummy */
  ) const
  {
    return thrust::make_tuple(v0_prop + v1_prop, static_cast<edge_t>(intersection.size()));
  }
};

}  // namespace

template <typename vertex_t, typename edge_t, bool multi_gpu>
void k_truss(raft::handle_t const& handle,
             graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
             edge_t k,
             bool do_expensive_check)
{
  using weight_t = float;  // dummy

  // 1. Check input arguments.

  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "Invalid input arguments: K-truss currently supports undirected graphs only.");
  CUGRAPH_EXPECTS(!graph_view.is_multigraph(),
                  "Invalid input arguments: K-truss currently does not support multi-graphs.");

  if (do_expensive_check) {
    // nothing to do
  }

  // 2. Exclude self-loops (FIXME: better mask-out once we add masking support).

  std::optional<graph_t<vertex_t, edge_t, false, multi_gpu>> modified_graph{std::nullopt};
  std::optional<graph_view_t<vertex_t, edge_t, false, multi_gpu>> modified_graph_view{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

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

  // 3. Find (k+1)-core and exclude edges that do not belong to (k+1)-core (FIXME: better mask-out
  // once we add masking support).
  // FIXME: Call k-core instead of core number
  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;
    auto vertex_partition_range_lasts =
      renumber_map
        ? std::make_optional<std::vector<vertex_t>>(cur_graph_view.vertex_partition_range_lasts())
        : std::nullopt;

    rmm::device_uvector<edge_t> core_numbers(cur_graph_view.local_vertex_partition_range_size(),
                                             handle.get_stream());
    core_number(handle,
                cur_graph_view,
                core_numbers.data(),
                k_core_degree_type_t::OUT,
                size_t{k-1},
                size_t{k-1});

    edge_src_property_t<decltype(cur_graph_view), bool> edge_src_in_k_plus_one_cores(
      handle, cur_graph_view);
    edge_dst_property_t<decltype(cur_graph_view), bool> edge_dst_in_k_plus_one_cores(
      handle, cur_graph_view);
    auto in_k_plus_one_core_first =
      thrust::make_transform_iterator(core_numbers.begin(), in_k_plus_one_or_greater_t<edge_t>{k});
    rmm::device_uvector<bool> in_k_plus_one_core_flags(core_numbers.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 in_k_plus_one_core_first,
                 in_k_plus_one_core_first + core_numbers.size(),
                 in_k_plus_one_core_flags.begin());
    update_edge_src_property(
      handle, cur_graph_view, in_k_plus_one_core_flags.begin(), edge_src_in_k_plus_one_cores);
    update_edge_dst_property(
      handle, cur_graph_view, in_k_plus_one_core_flags.begin(), edge_dst_in_k_plus_one_cores);
    auto [srcs, dsts] = extract_transform_e(handle,
                                            cur_graph_view,
                                            edge_src_in_k_plus_one_cores.view(),
                                            edge_dst_in_k_plus_one_cores.view(),
                                            edge_dummy_property_t{}.view(),
                                            extract_k_plus_one_core_t<vertex_t>{});

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t>(
          handle, std::move(srcs), std::move(dsts), std::nullopt, std::nullopt, std::nullopt);
    }

    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};

    std::tie(*modified_graph, std::ignore, std::ignore, std::ignore, tmp_renumber_map) =
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

    if (renumber_map) {  // collapse renumber_map
      unrenumber_int_vertices<vertex_t, multi_gpu>(handle,
                                                   (*tmp_renumber_map).data(),
                                                   (*tmp_renumber_map).size(),
                                                   (*renumber_map).data(),
                                                   *vertex_partition_range_lasts);
    }
    renumber_map = std::move(tmp_renumber_map);
  }


  // 4. Keep only the edges from a low-degree vertex to a high-degree vertex.
  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;


    // FIXME: REmove this**********************************************************************************
    rmm::device_uvector<vertex_t> srcs_(0, handle.get_stream());
    rmm::device_uvector<vertex_t> dsts_(0, handle.get_stream());

    std::tie(srcs_, dsts_, std::ignore, std::ignore) = decompress_to_edgelist(
      handle,
      cur_graph_view,
      std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
      std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
      std::optional<raft::device_span<vertex_t const>>(std::nullopt));
  
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
    auto [srcs, dsts] = extract_transform_e(handle,
                                            cur_graph_view,
                                            edge_src_out_degrees.view(),
                                            edge_dst_out_degrees.view(),
                                            edge_dummy_property_t{}.view(),
                                            extract_low_to_high_degree_edges_t<vertex_t, edge_t>{});

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t>(
          handle, std::move(srcs), std::move(dsts), std::nullopt, std::nullopt, std::nullopt);
    }

    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};
    std::tie(*modified_graph, std::ignore, std::ignore, std::ignore, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{false /* now asymmetric */, cur_graph_view.is_multigraph()},
        false); //******************FIXME:        hardcoded to False

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
    rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());

    std::tie(edgelist_srcs, edgelist_dsts, std::ignore, std::ignore) = decompress_to_edgelist(
      handle,
      cur_graph_view,
      std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
      std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
      std::optional<raft::device_span<vertex_t const>>(std::nullopt));

    auto edge_first =
      thrust::make_zip_iterator(edgelist_dsts.begin(), edgelist_srcs.begin());
    
    auto num_triangles =  edge_triangle_count<vertex_t, edge_t, false, false>(handle,
                                              cur_graph_view,
                                              raft::device_span<vertex_t>(edgelist_srcs.data(), edgelist_srcs.size()),
                                              raft::device_span<vertex_t>(edgelist_dsts.data(), edgelist_dsts.size()));
                            
  
    auto edge_triangle_count_pair_first = thrust::make_zip_iterator(
      edge_first, num_triangles.begin());
  
    // Note: ensure 'edges_with_triangles' and 'cur_graph_view' have the same transpose flag
    cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edges_with_triangles(handle);

  
    cugraph::edge_property_t<decltype(cur_graph_view), bool> edge_value_output(handle,
                                                                               cur_graph_view);
    
    edge_t num_valid_edges = edgelist_srcs.size();
    edge_t num_invalid_edges{0};
    size_t num_edges_with_triangles{0};

    while (true) {
    // Remove edges that have a triangle count of zero. Those should not be accounted
    // for during the unroling phase.
    auto edges_with_triangle_last =
      thrust::stable_partition(handle.get_thrust_policy(),
                               edge_triangle_count_pair_first,
                               edge_triangle_count_pair_first + num_triangles.size(),
                               [k] __device__(auto e) {
                                 auto num_triangles = thrust::get<1>(e);
                                 return num_triangles > 0;
                               });
    
    num_edges_with_triangles = static_cast<size_t>(
      thrust::distance(edge_triangle_count_pair_first, edges_with_triangle_last));
    

    
    edgelist_srcs.resize(num_edges_with_triangles, handle.get_stream());
    edgelist_dsts.resize(num_edges_with_triangles, handle.get_stream());
    num_triangles.resize(num_edges_with_triangles, handle.get_stream());

    // 'invalid_edge_first' marks the beginning of the edges to be removed
    auto invalid_edge_first =
      thrust::stable_partition(handle.get_thrust_policy(),
                               edge_triangle_count_pair_first,
                               edge_triangle_count_pair_first + num_triangles.size(),
                               [k] __device__(auto e) {
                                 auto num_triangles = thrust::get<1>(e);
                                 return num_triangles >= k - 2;
                               });
    
    num_invalid_edges = static_cast<size_t>(
      thrust::distance(invalid_edge_first, edge_triangle_count_pair_first + edgelist_srcs.size()));

    auto num_valid_edges = edgelist_srcs.size() - num_invalid_edges;
    if (num_invalid_edges == 0){
        break;
    }

    // copy invalid edges
    auto invalid_edges_buffer = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
      num_invalid_edges, handle.get_stream());
    


    
    thrust::copy(handle.get_thrust_policy(),
                 thrust::make_zip_iterator(edgelist_srcs.begin() + num_valid_edges, edgelist_dsts.begin() + num_valid_edges),
                 thrust::make_zip_iterator(edgelist_srcs.begin() + edgelist_srcs.size(), edgelist_dsts.begin() + edgelist_srcs.size()),
                 get_dataframe_buffer_begin(invalid_edges_buffer));
    
    // resize the 'edgelist_srcs' and 'edgelist_dst'
    edgelist_srcs.resize(num_valid_edges, handle.get_stream());
    edgelist_dsts.resize(num_valid_edges, handle.get_stream());
    num_triangles.resize(num_valid_edges, handle.get_stream());
    
      
    // sort back the edge as those are needed later when running a binary tree
    thrust::sort_by_key(handle.get_thrust_policy(),
                        edge_first,
                        edge_first + edgelist_srcs.size(),
                        num_triangles.begin());
    
      // case 2: unroll (q, r)
      // FIXME: Update the num_edges when removing edges
      // For each (q, r) edges to unroll, find the incoming edges to 'r' let's say from 'p' and
      // create the pair (p, q)
      rmm::device_uvector<vertex_t> prefix_sum(num_invalid_edges + 1, handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        prefix_sum.begin(),
        prefix_sum.end(),
        [invalid_first   = get_dataframe_buffer_begin(invalid_edges_buffer),
         //dst_array_begin = std::get<0>(incoming_vertex_pairs).begin(),
         dst_array_begin = edgelist_dsts.begin(),
         num_edges       = edgelist_srcs.size()] __device__(auto idx) {
          auto src           = thrust::get<0>(*(invalid_first + idx));
          auto dst           = thrust::get<1>(*(invalid_first + idx));
          auto dst_array_end = dst_array_begin + num_edges;
          auto itr_lower = thrust::lower_bound(thrust::seq, dst_array_begin, dst_array_end, dst);
          auto itr_upper = thrust::upper_bound(thrust::seq, itr_lower, dst_array_end, dst);
          auto dist      = thrust::distance(itr_lower, itr_upper);
          return dist;
        });
      thrust::exclusive_scan(
        handle.get_thrust_policy(), prefix_sum.begin(), prefix_sum.end(), prefix_sum.begin());

      auto vertex_pair_buffer_p_q = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        prefix_sum.back_element(handle.get_stream()), handle.get_stream());

      auto vertex_pair_buffer_p_r = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        prefix_sum.back_element(handle.get_stream()), handle.get_stream());

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(num_invalid_edges),
        [invalid_first_dst           = std::get<1>(invalid_edges_buffer).begin(),
         invalid_first_src           = std::get<0>(invalid_edges_buffer).begin(),
         dst_array_begin             = edgelist_dsts.begin(),
         prefix_sum                  = prefix_sum.data(),
         incoming_vertex_pairs       = edge_first,
         vertex_pair_buffer_p_q      = get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
         vertex_pair_buffer_p_r      = get_dataframe_buffer_begin(vertex_pair_buffer_p_r),
         num_edges                   = edgelist_srcs.size()] __device__(auto idx) {
          printf("\nidx =%d\n", idx);
          auto src             = invalid_first_src[idx];
          auto dst             = invalid_first_dst[idx];
          auto dst_array_end   = dst_array_begin + num_edges;

          auto itr_lower = thrust::lower_bound(thrust::seq, dst_array_begin, dst_array_end, dst);
          auto idx_lower = thrust::distance(
            dst_array_begin, itr_lower);  // Need a binary search to find the begining of the range
        
          thrust::tabulate(
            thrust::seq,
            vertex_pair_buffer_p_q + prefix_sum[idx],
            vertex_pair_buffer_p_q + prefix_sum[idx + 1],
            [incoming_vertex_pairs = incoming_vertex_pairs,
             dst                         = dst,
             src                         = src,
             idx_lower                   = idx_lower](auto idx_in_segment) {
              return thrust::make_tuple(
                thrust::get<1>(*(incoming_vertex_pairs + idx_lower + idx_in_segment)), src);
            });

          thrust::tabulate(
            thrust::seq,
            vertex_pair_buffer_p_r + prefix_sum[idx],
            vertex_pair_buffer_p_r + prefix_sum[idx + 1],
            [incoming_vertex_pairs = incoming_vertex_pairs,
             dst                         = dst,
             src                         = src,
             idx,
             idx_lower                   = idx_lower](auto idx_in_segment) {
              return thrust::make_tuple(
                thrust::get<1>(*(incoming_vertex_pairs + idx_lower + idx_in_segment)), dst);
            });
        });

      auto edges_exist = cur_graph_view.has_edge(
        handle,
        raft::device_span<vertex_t const>(std::get<0>(vertex_pair_buffer_p_q).data(),
                                          std::get<0>(vertex_pair_buffer_p_q).size()),
        raft::device_span<vertex_t const>(std::get<1>(vertex_pair_buffer_p_q).data(),
                                          std::get<1>(vertex_pair_buffer_p_q).size()));

      auto edge_to_existance = thrust::make_zip_iterator(
        thrust::make_zip_iterator(get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
                                  get_dataframe_buffer_begin(vertex_pair_buffer_p_r)),
        edges_exist.begin());

      auto has_edge_last = thrust::remove_if(handle.get_thrust_policy(),
                                             edge_to_existance,
                                             edge_to_existance + edges_exist.size(),
                                             [] __device__(auto e) {
                                             auto edge_exists = thrust::get<1>(e);
                                             return edge_exists == 0;
                                            });
      
      auto num_edge_exists = thrust::distance(edge_to_existance, has_edge_last);

      // After pushing the non-existant edges to the second partition,
      // remove them by resizing  both vertex pair buffer
      resize_dataframe_buffer(vertex_pair_buffer_p_q, num_edge_exists, handle.get_stream());
      resize_dataframe_buffer(vertex_pair_buffer_p_r, num_edge_exists, handle.get_stream());

      auto edge_last = edge_first + edgelist_srcs.size();
      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_counting_iterator<edge_t>(0),
                       thrust::make_counting_iterator<edge_t>(num_edge_exists),
                       unroll_edge<vertex_t, edge_t, decltype(edge_first)>{
                       raft::device_span<vertex_t>(num_triangles.data(), num_triangles.size()),
                       get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
                       edge_first,
                       edge_last,
                       });

      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_counting_iterator<edge_t>(0),
                       thrust::make_counting_iterator<edge_t>(num_edge_exists),
                       unroll_edge<vertex_t, edge_t, decltype(edge_first)>{
                       raft::device_span<vertex_t>(num_triangles.data(), num_triangles.size()),
                       get_dataframe_buffer_begin(vertex_pair_buffer_p_r),
                       edge_first,
                       edge_last,
                       });     
      
      // Put edges with triangle count == 0 in the second partition
      // FIXME: revisit all the 'stable_partition' and only used them
      // when necessary otherwise simply call 'thrust::partition'
      // Stable_parition is needed because we want to keep src and dst sorted
      // so that we don't need to sort it again.
      // FIXME: Create a rountine capturing L719:L763 as this block of code gets
      // repeated
      auto last_edge_with_triangles =
        thrust::stable_partition(handle.get_thrust_policy(),
                                 edge_triangle_count_pair_first,
                                 edge_triangle_count_pair_first + edgelist_srcs.size(),
                                 [] __device__(auto edge_to_num_triangles) {
                                   return thrust::get<1>(edge_to_num_triangles) > 0;
                                 });

      auto last_edge_with_triangles_idx = thrust::distance(edge_triangle_count_pair_first, last_edge_with_triangles);
         
      // Need to run prefix_sum again to get new ranges because some incoming edges were removed
      prefix_sum.resize(num_invalid_edges + 1, handle.get_stream());

      thrust::tabulate(
        handle.get_thrust_policy(),
        prefix_sum.begin(),
        prefix_sum.end(),
        [invalid_first               = get_dataframe_buffer_begin(invalid_edges_buffer),
         dst_array_begin = edgelist_dsts.begin(),
         num_edges                   = edgelist_srcs.size()] __device__(auto idx) {
          auto src = thrust::get<0>(*(invalid_first + idx));
          auto dst = thrust::get<1>(*(invalid_first + idx));
          auto dst_array_end = dst_array_begin + num_edges;
          auto itr_lower = thrust::lower_bound(thrust::seq, dst_array_begin, dst_array_end, dst);
          // FIXME: revisit dst_array_begin vs itr_lower
          auto itr_upper = thrust::upper_bound(thrust::seq, itr_lower, dst_array_end, dst);
          auto dist      = thrust::distance(itr_lower, itr_upper);
          return dist;
        });

      thrust::exclusive_scan(
        handle.get_thrust_policy(), prefix_sum.begin(), prefix_sum.end(), prefix_sum.begin());
      
      // case 3 unroll (p, r)
      vertex_pair_buffer_p_q = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        prefix_sum.back_element(handle.get_stream()), handle.get_stream());

      auto vertex_pair_buffer_q_r = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        prefix_sum.back_element(handle.get_stream()), handle.get_stream());

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(num_invalid_edges),
        [invalid_first_dst           = std::get<1>(invalid_edges_buffer).begin(),
         invalid_first_src           = std::get<0>(invalid_edges_buffer).begin(),
         dst_array_begin             = edgelist_dsts.begin(),
         prefix_sum                  = prefix_sum.data(),
         incoming_vertex_pairs       = edge_first,
         vertex_pair_buffer_p_q      = get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
         vertex_pair_buffer_q_r      = get_dataframe_buffer_begin(vertex_pair_buffer_q_r),
         num_edges                   = edgelist_srcs.size()] __device__(auto idx) {
          auto src             = invalid_first_src[idx];
          auto dst             = invalid_first_dst[idx];
          auto dst_array_end   = dst_array_begin + num_edges;
          auto itr_lower = thrust::lower_bound(thrust::seq, dst_array_begin, dst_array_end, dst);
          auto idx_lower = thrust::distance(
            dst_array_begin, itr_lower);  // Need a binary search to find the begining of the range

          thrust::tabulate(
            thrust::seq,
            vertex_pair_buffer_p_q + prefix_sum[idx],
            vertex_pair_buffer_p_q + prefix_sum[idx + 1],
            [incoming_vertex_pairs = incoming_vertex_pairs,
             dst                         = dst,
             src                         = src,
             idx_lower                   = idx_lower](auto idx_in_segment) {
              return thrust::make_tuple(
                src, thrust::get<1>(*(incoming_vertex_pairs + idx_lower + idx_in_segment)));
            });

          thrust::tabulate(
            thrust::seq,
            vertex_pair_buffer_q_r + prefix_sum[idx],
            vertex_pair_buffer_q_r + prefix_sum[idx + 1],
            [incoming_vertex_pairs = incoming_vertex_pairs,
             dst                         = dst,
             src                         = src,
             idx_lower                   = idx_lower](auto idx_in_segment) {
              return thrust::make_tuple(
                thrust::get<1>(*(incoming_vertex_pairs + idx_lower + idx_in_segment)), dst);
            });
        });

      edges_exist = cur_graph_view.has_edge(
        handle,
        raft::device_span<vertex_t const>(std::get<0>(vertex_pair_buffer_p_q).data(),
                                          std::get<0>(vertex_pair_buffer_p_q).size()),
        raft::device_span<vertex_t const>(std::get<1>(vertex_pair_buffer_p_q).data(),
                                          std::get<1>(vertex_pair_buffer_p_q).size()));

      edge_to_existance = thrust::make_zip_iterator(
        thrust::make_zip_iterator(get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
                                  get_dataframe_buffer_begin(vertex_pair_buffer_q_r)),
        edges_exist.begin());
      
      has_edge_last = thrust::remove_if(handle.get_thrust_policy(),
                                             edge_to_existance,
                                             edge_to_existance + edges_exist.size(),
                                             [] __device__(auto e) {
                                             auto edge_exists = thrust::get<1>(e);
                                             return edge_exists == 0;
                                            });
      
      num_edge_exists = thrust::distance(edge_to_existance, has_edge_last);

      // After pushing the non-existant edges to the second partition,
      // remove them by resizing  both vertex pair buffer
      resize_dataframe_buffer(vertex_pair_buffer_p_q, num_edge_exists, handle.get_stream());
      resize_dataframe_buffer(vertex_pair_buffer_q_r, num_edge_exists, handle.get_stream());
      edges_exist.resize(size_dataframe_buffer(vertex_pair_buffer_p_q), handle.get_stream());

      edge_last = edge_first + edgelist_srcs.size();
      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_counting_iterator<edge_t>(0),
                       thrust::make_counting_iterator<edge_t>(num_edge_exists),
                       unroll_edge<vertex_t, edge_t, decltype(edge_first)>{
                       raft::device_span<vertex_t>(num_triangles.data(), num_triangles.size()),
                       get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
                       edge_first,
                       edge_last,
                       });

      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_counting_iterator<edge_t>(0),
                       thrust::make_counting_iterator<edge_t>(num_edge_exists),
                       unroll_edge<vertex_t, edge_t, decltype(edge_first)>{
                       raft::device_span<vertex_t>(num_triangles.data(), num_triangles.size()),
                       get_dataframe_buffer_begin(vertex_pair_buffer_q_r),
                       edge_first,
                       edge_last,
                       });     

      // Put edges with triangle count == 0 in the second partition
      // FIXME: revisit all the 'stable_partition' and only used them
      // when necessary otherwise simply call 'thrust::partition'
      // Stable_parition is needed because we want to keep src and dst sorted
      last_edge_with_triangles =
        thrust::stable_partition(handle.get_thrust_policy(),
                                 edge_triangle_count_pair_first,
                                 edge_triangle_count_pair_first + edgelist_srcs.size(),
                                 [] __device__(auto edge_to_num_triangles) {
                                   return thrust::get<1>(edge_to_num_triangles) > 0;
                                 });

      last_edge_with_triangles_idx = thrust::distance(edge_triangle_count_pair_first, last_edge_with_triangles);

   

      // resize the 'edgelist_srcs' and 'edgelsit_dst'
      edgelist_srcs.resize(last_edge_with_triangles_idx, handle.get_stream());
      edgelist_dsts.resize(last_edge_with_triangles_idx, handle.get_stream());
      num_triangles.resize(last_edge_with_triangles_idx, handle.get_stream());
      
      // rename the above it to last_edge_with_triangles
      // rename the below to edges_with_triangles
      edges_with_triangles.clear();  // FIXME: is this needed?

      edges_with_triangles.insert(edgelist_srcs.begin(),
                       edgelist_srcs.begin(),
                       edgelist_dsts.end());                  

      cugraph::transform_e(
        handle,
        cur_graph_view,
        edges_with_triangles,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {
          return true;
        },
        edge_value_output.mutable_view(),
        false);

      cur_graph_view.attach_edge_mask(edge_value_output.view());

      // case 1. For the (p, q), find intersection 'r' to create (p, r, -1) and (q, r, -1)
      // FIXME: check if 'invalid_edge_first' is necessery as I operate on 'vertex_pair_buffer'
      // which contains the ordering with the number of triangles.
      // FIXME: debug this stage. There are edges that have been removed that are still found in nbr intersection
      auto [intersection_offsets, intersection_indices] =
        detail::nbr_intersection(handle,
                                 cur_graph_view,
                                 cugraph::edge_dummy_property_t{}.view(),
                                 get_dataframe_buffer_begin(invalid_edges_buffer),
                                 get_dataframe_buffer_end(invalid_edges_buffer),
                                 std::array<bool, 2>{true, true},
                                 do_expensive_check);

      size_t accumulate_pair_size =
        intersection_indices.size();
      
      auto vertex_pair_buffer_p_r_edge_p_q =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          accumulate_pair_size, handle.get_stream());
      
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_p_q),
        get_dataframe_buffer_end(vertex_pair_buffer_p_r_edge_p_q),
          generate_p_r<vertex_t, edge_t, decltype(get_dataframe_buffer_begin(invalid_edges_buffer))>{
            raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
            raft::device_span<vertex_t const>(intersection_indices.data(),
                                              intersection_indices.size()),
            get_dataframe_buffer_begin(invalid_edges_buffer)
          });
        
      edge_last = edge_first + edgelist_srcs.size();
      num_edge_exists = accumulate_pair_size;
      thrust::for_each(handle.get_thrust_policy(),
                      thrust::make_counting_iterator<edge_t>(0),
                      thrust::make_counting_iterator<edge_t>(num_edge_exists),
                      unroll_edge<vertex_t, edge_t, decltype(edge_first)>{
                      raft::device_span<vertex_t>(num_triangles.data(), num_triangles.size()),
                      get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_p_q),
                      edge_first,
                      edge_last,
                      });

      auto vertex_pair_buffer_q_r_edge_p_q =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          accumulate_pair_size, handle.get_stream());

      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q),
        get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q) +
          accumulate_pair_size,
        generate_q_r<vertex_t, edge_t, decltype(get_dataframe_buffer_begin(invalid_edges_buffer))>{
          raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          get_dataframe_buffer_begin(invalid_edges_buffer)  // FIXME: verify this is accurate
        });
      
      thrust::for_each(handle.get_thrust_policy(),
                      thrust::make_counting_iterator<edge_t>(0),
                      thrust::make_counting_iterator<edge_t>(num_edge_exists),
                      unroll_edge<vertex_t, edge_t, decltype(edge_first)>{
                      raft::device_span<vertex_t>(num_triangles.data(), num_triangles.size()),
                      get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q),
                      edge_first,
                      edge_last,
                      });
        
      // Put edges with triangle count == 0 in the second partition
      // FIXME: revisit all the 'stable_partition' and only used them
      // when necessary otherwise simply call 'thrust::partition'
      // Stable_parition is needed because we want to keep src and dst sorted
      // so that we don't need to sort it again.
      last_edge_with_triangles =
        thrust::stable_partition(handle.get_thrust_policy(),
                                 edge_triangle_count_pair_first,
                                 edge_triangle_count_pair_first + edgelist_srcs.size(),
                                 [] __device__(auto edge_to_num_triangles) {
                                   return thrust::get<1>(edge_to_num_triangles) > 0;
                                 });


      last_edge_with_triangles_idx = thrust::distance(edge_triangle_count_pair_first, last_edge_with_triangles);
      // rename the above it to last_edge_with_triangles

      // resize the 'edgelist_srcs' and 'edgelsit_dst'
      edgelist_srcs.resize(last_edge_with_triangles_idx, handle.get_stream());
      edgelist_dsts.resize(last_edge_with_triangles_idx, handle.get_stream());
      num_triangles.resize(last_edge_with_triangles_idx, handle.get_stream());

      printf("\n*******final*******\n");
      raft::print_device_vector("srcs", edgelist_srcs.data(), edgelist_srcs.size(), std::cout);
      raft::print_device_vector("dsts", edgelist_dsts.data(), edgelist_dsts.size(), std::cout);
      raft::print_device_vector("n_tr", num_triangles.data(), num_triangles.size(), std::cout);

      auto invalid_edge_last_ =
      thrust::stable_partition(handle.get_thrust_policy(),
                               edge_triangle_count_pair_first,
                               edge_triangle_count_pair_first + num_triangles.size(),
                               [k] __device__(auto e) {
                                 auto num_triangles = thrust::get<1>(e);
                                 return num_triangles < k - 2;
                               });
    
      num_invalid_edges = static_cast<size_t>(
        thrust::distance(edge_triangle_count_pair_first, invalid_edge_last_));


      // copy invalid edges
      resize_dataframe_buffer(invalid_edges_buffer, num_invalid_edges, handle.get_stream());
      
      thrust::copy(handle.get_thrust_policy(),
                  edge_first,
                  edge_first + num_invalid_edges,
                  get_dataframe_buffer_begin(invalid_edges_buffer));
      
      // sort back the edges as those are needed later when running a binary tree
      thrust::sort_by_key(handle.get_thrust_policy(),
                          edge_first,
                          edge_first + edgelist_srcs.size(),
                          num_triangles.begin());

    }

    if (num_invalid_edges == edgelist_srcs.size()) {
      // return empty graph view
      std::optional<graph_view_t<vertex_t, edge_t, false, multi_gpu>> empty_graph_view{std::nullopt};
      printf("\nreturning an empty graph");
      // FIXME: To be updated
      // return empty_graph_view;
    }
    else{
      // FIXME: To be updated
      // return cur_graph_view;
      printf("\nreturning a non empty graph with num_edges = %d\n", edgelist_srcs.size());
    }
  
    
  
  }
}

}  // namespace cugraph
