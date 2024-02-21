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
  __device__ bool operator()(edge_t core_number) const { return core_number >= k + 1; }
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

template <typename vertex_t, typename edge_t, typename VertexPairIterator>
struct extract_p_q {
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  raft::device_span<vertex_t> num_triangles{};

  VertexPairIterator edges{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);
    unsigned int r = atomicInc((unsigned int*)(num_triangles.data()+idx), (unsigned int)1);
  }
};


template <typename vertex_t, typename edge_t, typename VertexPairIterator>
struct extract_p_r {
  size_t num_vertex_pair{}; // rename to num_edges
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  raft::device_span<vertex_t> num_triangles{};

  VertexPairIterator edges{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);
    auto p_r_pair =
      thrust::make_tuple(thrust::get<0>(*(edges + idx)), intersection_indices[i]);
    // Find its position in 'edges'
    auto itr_p_r = thrust::lower_bound(thrust::seq,
                                       edges,
                                       edges + num_vertex_pair, // pass the number of vertex pairs
                                       p_r_pair);
    
    idx = thrust::distance(edges, itr_p_r);
    auto r = atomicAdd(num_triangles.data()+idx, 1);
    
  }
};


template <typename vertex_t, typename edge_t, typename VertexPairIterator>
struct extract_q_r {
  size_t num_vertex_pair; // rename to num_edges
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  raft::device_span<vertex_t> num_triangles{};

  VertexPairIterator edges{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);
    auto q_r_pair =
      thrust::make_tuple(thrust::get<1>(*(edges + idx)), intersection_indices[i]);
    
    // Find its position in 'edges'
    auto itr_q_r = thrust::lower_bound(thrust::seq,
                                       edges,
                                       edges + num_vertex_pair,
                                       q_r_pair);
    
    idx = thrust::distance(edges, itr_q_r);
    auto r = atomicAdd(num_triangles.data()+idx, 1);
    
  }
};


template <typename vertex_t, typename edge_t, typename VertexPairIterator>
struct unroll_edge {
  raft::device_span<vertex_t> num_triangles{};
  VertexPairIterator edge_unrolled{};
  VertexPairIterator edges{};
  VertexPairIterator edges_last{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto pair =
      thrust::make_tuple(thrust::get<0>(*(edge_unrolled + i)), thrust::get<1>(*(edge_unrolled + i)));        
    // Find its position in 'edges'
    auto itr = thrust::lower_bound(thrust::seq,
                                   edges,
                                   //edges + 6, // pass the number of vertex pairs
                                   edges_last,
                                   //thrust::make_tuple(0, 3)
                                   pair
                                   );

    auto idx = thrust::distance(edges, itr);
    auto r = atomicAdd(num_triangles.data() + idx, -1);
    
  }
};


template <typename vertex_t, typename edge_t, typename VertexPairIterator>
struct generate_p_r {
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};

  VertexPairIterator edges{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);
    auto pair =
      thrust::make_tuple(thrust::get<0>(*(edges + idx)), intersection_indices[i]);

    return pair;
  }
};



template <typename vertex_t, typename edge_t, typename VertexPairIterator>
struct generate_q_r {
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};

  VertexPairIterator edges{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);
    auto pair =
      thrust::make_tuple(thrust::get<1>(*(edges + idx)), intersection_indices[i]);

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
                size_t{k + 1},
                size_t{k + 1});

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
    std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
    rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());

    std::tie(edgelist_srcs, edgelist_dsts, std::ignore, std::ignore) = decompress_to_edgelist(
      handle,
      cur_graph_view,
      std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
      std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
      std::optional<raft::device_span<vertex_t const>>(std::nullopt));

    auto edges =
      thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());

    thrust::sort(
      handle.get_thrust_policy(), edges, edges + edgelist_srcs.size());

    size_t num_edges = edgelist_srcs.size(); // FIXME: rename to num_edges and always update values when removing edges

    // FIXME: Perform nbr_intersection in chuncks.
    auto [intersection_offsets, intersection_indices] =
      detail::nbr_intersection(handle,
                               cur_graph_view,
                               cugraph::edge_dummy_property_t{}.view(),
                               edges,
                               edges + num_edges,
                               std::array<bool, 2>{true, true},
                               do_expensive_check);
  
    rmm::device_uvector<vertex_t> num_triangles(num_edges, handle.get_stream());
    thrust::fill(
      handle.get_thrust_policy(), num_triangles.begin(), num_triangles.end(), size_t{0});

    // Update the number of triangles of each (p, q) edges by looking at their intersection
    // size
    thrust::adjacent_difference(handle.get_thrust_policy(),
                                intersection_offsets.begin() + 1,
                                intersection_offsets.end(),
                                num_triangles.begin());

    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator<edge_t>(0),
                     thrust::make_counting_iterator<edge_t>(intersection_indices.size()),
                     extract_p_r<vertex_t, edge_t, decltype(edges)>{
                     num_edges,
                     raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
                     raft::device_span<vertex_t const>(intersection_indices.data(), intersection_indices.size()),
                     raft::device_span<vertex_t>(num_triangles.data(), num_triangles.size()),
                     edges});
    
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator<edge_t>(0),
                     thrust::make_counting_iterator<edge_t>(intersection_indices.size()),
                     extract_q_r<vertex_t, edge_t, decltype(edges)>{
                     num_edges,
                     raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
                     raft::device_span<vertex_t const>(intersection_indices.data(), intersection_indices.size()),
                     raft::device_span<vertex_t>(num_triangles.data(), num_triangles.size()),
                     edges});

    auto edges_to_num_triangles = thrust::make_zip_iterator(
      edges, num_triangles.begin());

    // 'invalid_edge_first' marks the beginning of the edges to be removed

    auto invalid_edge_last =
      thrust::stable_partition(handle.get_thrust_policy(),
                               edges_to_num_triangles,
                               edges_to_num_triangles + num_triangles.size(),
                               [k] __device__(auto e) {
                                 auto num_triangles = thrust::get<1>(e);
                                 return num_triangles < k - 2;
                               });
    
    size_t num_invalid_edges{0};
    num_invalid_edges = static_cast<size_t>(
      thrust::distance(edges_to_num_triangles, invalid_edge_last));


    // copy invalid edges
    auto invalid_edges_buffer = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
      num_invalid_edges, handle.get_stream());
    
    thrust::copy(handle.get_thrust_policy(),
                 edges,
                 edges + num_invalid_edges,
                 get_dataframe_buffer_begin(invalid_edges_buffer));
    
    // sort back the edges as those are needed later when running a binary tree
    thrust::sort_by_key(handle.get_thrust_policy(),
                        edges,
                        edges + num_edges,
                        num_triangles.begin());


    // Unroll and remove/mask edges as long as there are still edges part
    // of the K-Truss.
    auto num_valid_edges = num_edges - num_invalid_edges;
    while ((num_valid_edges != 0) && (num_invalid_edges !=0)) {

      // case 2: unroll (q, r)
      // FIXME: Update the num_edges when removing edges

      
      // FIXME: Need a buffer for the incomming vertex pairs because 'edges_to_num_triangles'
      // is sorted in a way that matches the number of triangles per edges therefore,
      // can't use a zip iterator for 'incoming_vertex_pairs'. This adds increase memory
      // footprint
      auto incoming_vertex_pairs = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        num_edges, handle.get_stream());

      thrust::tabulate(handle.get_thrust_policy(),
                       get_dataframe_buffer_begin(incoming_vertex_pairs),
                       get_dataframe_buffer_end(incoming_vertex_pairs),
                       [edges=edges
                       ]
                       __device__(edge_t idx){
                          auto edge = thrust::make_tuple(thrust::get<1>(*(edges + idx)), thrust::get<0>(*(edges + idx)));
                          return edge;
                          
                       });
      
      // Sort the 'incoming_vertex_pairs' by 'dst'
      thrust::sort(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(incoming_vertex_pairs),
        get_dataframe_buffer_end(incoming_vertex_pairs));

      // For each (q, r) edges to unroll, find the incoming edges to 'r' let's say from 'p' and
      // create the pair (p, q)
      rmm::device_uvector<int> prefix_sum(num_invalid_edges + 1, handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        prefix_sum.begin(),
        prefix_sum.end(),
        [invalid_first   = get_dataframe_buffer_begin(invalid_edges_buffer),
         dst_array_begin = std::get<0>(incoming_vertex_pairs).begin(),
         num_edges       = num_edges] __device__(auto idx) {
          auto src           = thrust::get<0>(*(invalid_first + idx));
          auto dst           = thrust::get<1>(*(invalid_first + idx));
          auto dst_array_end = dst_array_begin + num_edges;
          auto itr_lower = thrust::lower_bound(thrust::seq, dst_array_begin, dst_array_end, dst);
          auto idx_lower =
            thrust::distance(dst_array_begin, itr_lower);  // FIXME: remove self loops
          auto itr_upper = thrust::upper_bound(thrust::seq, dst_array_begin, dst_array_end, dst);
          auto idx_upper = thrust::distance(dst_array_begin, itr_upper);
          auto dist      = thrust::distance(itr_lower, itr_upper);
          return dist;
        });
      thrust::exclusive_scan(
        handle.get_thrust_policy(), prefix_sum.begin(), prefix_sum.end(), prefix_sum.begin());
      
      auto vertex_pair_buffer_p_q = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        prefix_sum.back_element(handle.get_stream()), handle.get_stream());

      auto vertex_pair_buffer_p_r = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        prefix_sum.back_element(handle.get_stream()), handle.get_stream());

      rmm::device_uvector<vertex_t> indices(num_edges, handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(), indices.begin(), indices.end(), thrust::identity<vertex_t>());

      thrust::for_each(
        handle.get_thrust_policy(),
        indices.begin(),
        indices.end(),
        [invalid_first_dst           = std::get<1>(invalid_edges_buffer).begin(),
         invalid_first_src           = std::get<0>(invalid_edges_buffer).begin(),
         prefix_sum                  = prefix_sum.data(),
         incoming_vertex_pairs = get_dataframe_buffer_begin(incoming_vertex_pairs),
         vertex_pair_buffer_p_q      = get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
         vertex_pair_buffer_p_r      = get_dataframe_buffer_begin(vertex_pair_buffer_p_r),
         num_edges                   = num_edges] __device__(auto idx) {
          auto src             = invalid_first_src[idx];
          auto dst             = invalid_first_dst[idx];
          auto dst_array_begin = invalid_first_dst;
          auto dst_array_end   = invalid_first_dst + num_edges;
          auto itr_lower = thrust::lower_bound(thrust::seq, dst_array_begin, dst_array_end, dst);
          auto idx_lower = thrust::distance(
            dst_array_begin, itr_lower);  // Need a binary search to find the begining of the range

          auto dist = prefix_sum[idx + 1] - prefix_sum[idx];

          thrust::tabulate(
            thrust::seq,
            vertex_pair_buffer_p_q + prefix_sum[idx],
            vertex_pair_buffer_p_q + prefix_sum[idx] + dist,
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
            vertex_pair_buffer_p_r + prefix_sum[idx] + dist,
            [incoming_vertex_pairs = incoming_vertex_pairs,
             dst                         = dst,
             src                         = src,
             idx_lower                   = idx_lower](auto idx_in_segment) {
              return thrust::make_tuple(
                thrust::get<1>(*(incoming_vertex_pairs + idx_lower + idx_in_segment)), dst);
            });
        });

      auto edge_exists = cur_graph_view.has_edge(
        handle,
        raft::device_span<vertex_t const>(std::get<0>(vertex_pair_buffer_p_q).data(),
                                          std::get<0>(vertex_pair_buffer_p_q).size()),
        raft::device_span<vertex_t const>(std::get<1>(vertex_pair_buffer_p_q).data(),
                                          std::get<1>(vertex_pair_buffer_p_q).size()));

      auto edge_to_existance = thrust::make_zip_iterator(
        thrust::make_zip_iterator(get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
                                  get_dataframe_buffer_begin(vertex_pair_buffer_p_r)),
        edge_exists.begin());

      auto has_edge_last = thrust::partition(handle.get_thrust_policy(),
                                                    edge_to_existance,
                                                    edge_to_existance + edge_exists.size(),
                                                    [] __device__(auto e) {
                                                      auto edge_exists = thrust::get<1>(e);
                                                      return edge_exists;
                                                    });

      auto num_edge_exists = thrust::distance(edge_to_existance, has_edge_last);

      // After pushing the non-existant edges to the second partition,
      // remove them by resizing  both vertex pair buffer
      resize_dataframe_buffer(vertex_pair_buffer_p_q, num_edge_exists, handle.get_stream());
      resize_dataframe_buffer(vertex_pair_buffer_p_r, num_edge_exists, handle.get_stream());

      auto edges_last = edges + num_edges;
      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_counting_iterator<edge_t>(0),
                       thrust::make_counting_iterator<edge_t>(num_edge_exists),
                       unroll_edge<vertex_t, edge_t, decltype(edges)>{
                       raft::device_span<vertex_t>(num_triangles.data(), num_triangles.size()),
                       get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
                       edges,
                       edges_last,
                       });

      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_counting_iterator<edge_t>(0),
                       thrust::make_counting_iterator<edge_t>(num_edge_exists),
                       unroll_edge<vertex_t, edge_t, decltype(edges)>{
                       raft::device_span<vertex_t>(num_triangles.data(), num_triangles.size()),
                       get_dataframe_buffer_begin(vertex_pair_buffer_p_r),
                       edges,
                       edges_last,
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
                                 edges_to_num_triangles,
                                 edges_to_num_triangles + num_edges,
                                 [] __device__(auto edge_to_num_triangles) {
                                   return thrust::get<1>(edge_to_num_triangles) > 0;
                                 });

      auto last_edge_with_triangles_idx = thrust::distance(edges_to_num_triangles, last_edge_with_triangles);
      // rename the above it to last_edge_with_triangles

      // Note: ensure 'edges_with_triangles' and 'cur_graph_view' have the same transpose flag
      cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edges_with_triangles(handle);

      cugraph::edge_property_t<decltype(cur_graph_view), bool> edge_value_output(handle,
                                                                                 cur_graph_view);

      // rename the below to edges_with_triangles
      edges_with_triangles.insert(edgelist_srcs.begin(),
                                  edgelist_srcs.begin() + last_edge_with_triangles_idx,
                                  edgelist_dsts.begin());                       

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

      // resize the 'edgelist_srcs' and 'edgelsit_dst'
      edgelist_srcs.resize(last_edge_with_triangles_idx, handle.get_stream());
      edgelist_dsts.resize(last_edge_with_triangles_idx, handle.get_stream());
      num_triangles.resize(last_edge_with_triangles_idx, handle.get_stream());

      num_edges = edgelist_srcs.size();

      resize_dataframe_buffer(
        incoming_vertex_pairs, num_edges, handle.get_stream());

      thrust::tabulate(handle.get_thrust_policy(),
                       get_dataframe_buffer_begin(incoming_vertex_pairs),
                       get_dataframe_buffer_end(incoming_vertex_pairs),
                       [edges=edges
                       ]
                       __device__(edge_t idx){
                          auto edge = thrust::make_tuple(thrust::get<1>(*(edges + idx)), thrust::get<0>(*(edges + idx)));
                          return edge;
                       });
      
      thrust::sort(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(incoming_vertex_pairs),
        get_dataframe_buffer_end(incoming_vertex_pairs));  // FIXME: No need to partition

      // FIXME: Among the invalid edges, identify those that were removed to
      // avoid extra panalization. One way to achieve it is by calling thrust::set_intersection
      // to filter out the removed edges. However this will require another array.
      
      // Find the intersection of 'invalid_edges_buffer' and 'edges' to extract the remaining invalid
      // edges that still need to be processed. Didn't used thrust::set_intersection because I didn't
      // want to create a temporary array
      auto invalid_edge_last =
        thrust::partition(handle.get_thrust_policy(),
                          get_dataframe_buffer_begin(invalid_edges_buffer),
                          get_dataframe_buffer_end(invalid_edges_buffer),
                          [edge_first = edges, // rename to 'edges'
                           edge_last = edges + num_edges,
                           num_edges = num_edges]
                           __device__(auto invalid_edge) {
                            auto itr = thrust::find(thrust::seq, edge_first, edge_last, invalid_edge);
                            auto idx = thrust::distance(edge_first, itr);
                            return idx < num_edges;
                          });
      
      num_invalid_edges = thrust::distance(get_dataframe_buffer_begin(invalid_edges_buffer), invalid_edge_last);


      resize_dataframe_buffer(
        invalid_edges_buffer, num_edges, handle.get_stream());

      // Need to run prefix_sum again to get new ranges because some incoming edges were removed
      prefix_sum.resize(num_edges + 1, handle.get_stream());

      // FIXME: need to sort 'incoming_vertex_pairs'. No need because a stable partition was
      // performed that preserve the sorting
      thrust::tabulate(
        handle.get_thrust_policy(),
        prefix_sum.begin(),
        prefix_sum.end(),
        [invalid_first               = get_dataframe_buffer_begin(invalid_edges_buffer),
         dst_array_begin = std::get<0>(incoming_vertex_pairs).begin(),
         num_edges                   = num_edges] __device__(auto idx) {
          auto src = thrust::get<0>(*(invalid_first + idx));
          auto dst = thrust::get<1>(*(invalid_first + idx));
          auto dst_array_end = dst_array_begin + num_edges;
          auto itr_lower = thrust::lower_bound(thrust::seq, dst_array_begin, dst_array_end, dst);
          auto idx_lower =
            thrust::distance(dst_array_begin, itr_lower);  // FIXME: remove self loops
          auto itr_upper = thrust::upper_bound(thrust::seq, dst_array_begin, dst_array_end, dst);
          auto idx_upper = thrust::distance(dst_array_begin, itr_upper);
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
        indices.begin(),
        indices.end(),
        [invalid_first_dst           = std::get<1>(invalid_edges_buffer).begin(),
         invalid_first_src           = std::get<0>(invalid_edges_buffer).begin(),
         prefix_sum                  = prefix_sum.data(),
         incoming_vertex_pairs = get_dataframe_buffer_begin(incoming_vertex_pairs),
         vertex_pair_buffer_p_q      = get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
         vertex_pair_buffer_q_r      = get_dataframe_buffer_begin(vertex_pair_buffer_q_r),
         num_edges                   = num_edges] __device__(auto idx) {
          auto src             = invalid_first_src[idx];
          auto dst             = invalid_first_dst[idx];
          auto dst_array_begin = invalid_first_dst;
          auto dst_array_end   = invalid_first_dst + num_edges;
          auto itr_lower = thrust::lower_bound(thrust::seq, dst_array_begin, dst_array_end, dst);
          auto idx_lower = thrust::distance(
            dst_array_begin, itr_lower);  // Need a binary search to find the begining of the range
          auto dist = prefix_sum[idx + 1] - prefix_sum[idx];

          thrust::tabulate(
            thrust::seq,
            vertex_pair_buffer_p_q + prefix_sum[idx],
            vertex_pair_buffer_p_q + prefix_sum[idx] + dist,
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
            vertex_pair_buffer_q_r + prefix_sum[idx] + dist,
            [incoming_vertex_pairs = incoming_vertex_pairs,
             dst                         = dst,
             src                         = src,
             idx_lower                   = idx_lower](auto idx_in_segment) {
              return thrust::make_tuple(
                thrust::get<1>(*(incoming_vertex_pairs + idx_lower + idx_in_segment)), dst);
            });
        });

      edge_exists = cur_graph_view.has_edge(
        handle,
        raft::device_span<vertex_t const>(std::get<0>(vertex_pair_buffer_p_q).data(),
                                          std::get<0>(vertex_pair_buffer_p_q).size()),
        raft::device_span<vertex_t const>(std::get<1>(vertex_pair_buffer_p_q).data(),
                                          std::get<1>(vertex_pair_buffer_p_q).size()));

      edge_to_existance = thrust::make_zip_iterator(
        thrust::make_zip_iterator(get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
                                  get_dataframe_buffer_begin(vertex_pair_buffer_q_r)),
        edge_exists.begin());
      
      has_edge_last = thrust::stable_partition(handle.get_thrust_policy(),
                                               edge_to_existance,
                                               edge_to_existance + edge_exists.size(),
                                               [] __device__(auto e) {
                                                auto edge_exists = thrust::get<1>(e);
                                                return edge_exists;
                                                });

      num_edge_exists = thrust::distance(edge_to_existance, has_edge_last);

      // After pushing the non-existant edges to the second partition,
      // remove them by resizing  both vertex pair buffer
      resize_dataframe_buffer(vertex_pair_buffer_p_q, num_edge_exists, handle.get_stream());
      resize_dataframe_buffer(vertex_pair_buffer_q_r, num_edge_exists, handle.get_stream());

      edges_last = edges + num_edges;
      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_counting_iterator<edge_t>(0),
                       thrust::make_counting_iterator<edge_t>(num_edge_exists),
                       unroll_edge<vertex_t, edge_t, decltype(edges)>{
                       raft::device_span<vertex_t>(num_triangles.data(), num_triangles.size()),
                       get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
                       edges,
                       edges_last,
                       });

      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_counting_iterator<edge_t>(0),
                       thrust::make_counting_iterator<edge_t>(num_edge_exists),
                       unroll_edge<vertex_t, edge_t, decltype(edges)>{
                       raft::device_span<vertex_t>(num_triangles.data(), num_triangles.size()),
                       get_dataframe_buffer_begin(vertex_pair_buffer_q_r),
                       edges,
                       edges_last,
                       });     

      // Put edges with triangle count == 0 in the second partition
      // FIXME: revisit all the 'stable_partition' and only used them
      // when necessary otherwise simply call 'thrust::partition'
      // Stable_parition is needed because we want to keep src and dst sorted
      last_edge_with_triangles =
        thrust::stable_partition(handle.get_thrust_policy(),
                                 edges_to_num_triangles,
                                 edges_to_num_triangles + num_edges,
                                 [] __device__(auto edge_to_num_triangles) {
                                   return thrust::get<1>(edge_to_num_triangles) > 0;
                                 });

      last_edge_with_triangles_idx = thrust::distance(edges_to_num_triangles, last_edge_with_triangles);
      // rename the above it to last_edge_with_triangles
      // rename the below to edges_with_triangles
      edges_with_triangles.clear();  // FIXME: is this needed?

      cugraph::edge_property_t<decltype(cur_graph_view), bool> edge_value_output_p_r(
        handle, cur_graph_view);
      edges_with_triangles.insert(edgelist_srcs.begin(),
                       edgelist_srcs.begin() + last_edge_with_triangles_idx,
                       edgelist_dsts.begin());                       

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
        edge_value_output_p_r.mutable_view(),
        false);

      cur_graph_view.attach_edge_mask(edge_value_output_p_r.view());

      // resize the 'edgelist_srcs' and 'edgelsit_dst'
      edgelist_srcs.resize(last_edge_with_triangles_idx, handle.get_stream());
      edgelist_dsts.resize(last_edge_with_triangles_idx, handle.get_stream());
      num_triangles.resize(last_edge_with_triangles_idx, handle.get_stream());

      num_edges = edgelist_srcs.size();

      // case 1. For the (p, q), find intersection 'r' to create (p, r, -1) and (q, r, -1)
      // FIXME: check if 'invalid_edge_first' is necessery as I operate on 'vertex_pair_buffer'
      // which contains the ordering with the number of triangles.
      invalid_edge_last =
        thrust::partition(handle.get_thrust_policy(),
                          get_dataframe_buffer_begin(invalid_edges_buffer),
                          get_dataframe_buffer_end(invalid_edges_buffer),
                          [edge_first = edges, // rename to 'edges'
                           edge_last = edges + num_edges,
                           num_edges = num_edges]
                           __device__(auto invalid_edge) {

                            auto itr = thrust::find(thrust::seq, edge_first, edge_last, invalid_edge);
                            auto idx = thrust::distance(edge_first, itr);
                            return idx < num_edges;
                          });
      
      num_invalid_edges = thrust::distance(get_dataframe_buffer_begin(invalid_edges_buffer), invalid_edge_last);

      resize_dataframe_buffer(
        invalid_edges_buffer, num_edges, handle.get_stream());
    
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
        
      edges_last = edges + num_edges;
      num_edge_exists = accumulate_pair_size;
      thrust::for_each(handle.get_thrust_policy(),
                      thrust::make_counting_iterator<edge_t>(0),
                      thrust::make_counting_iterator<edge_t>(num_edge_exists),
                      unroll_edge<vertex_t, edge_t, decltype(edges)>{
                      raft::device_span<vertex_t>(num_triangles.data(), num_triangles.size()),
                      get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_p_q),
                      edges,
                      edges_last,
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
                      unroll_edge<vertex_t, edge_t, decltype(edges)>{
                      raft::device_span<vertex_t>(num_triangles.data(), num_triangles.size()),
                      get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q),
                      edges,
                      edges_last,
                      });
        
      // Put edges with triangle count == 0 in the second partition
      // FIXME: revisit all the 'stable_partition' and only used them
      // when necessary otherwise simply call 'thrust::partition'
      // Stable_parition is needed because we want to keep src and dst sorted
      // so that we don't need to sort it again.
      last_edge_with_triangles =
        thrust::stable_partition(handle.get_thrust_policy(),
                                 edges_to_num_triangles,
                                 edges_to_num_triangles + num_edges,
                                 [] __device__(auto edge_to_num_triangles) {
                                   return thrust::get<1>(edge_to_num_triangles) > 0;
                                 });

      last_edge_with_triangles_idx = thrust::distance(edges_to_num_triangles, last_edge_with_triangles);
      // rename the above it to last_edge_with_triangles

      // Note: ensure 'edges_with_triangles' and 'cur_graph_view' have the same transpose flag
      edges_with_triangles.clear();  // FIXME: is this needed?

      cugraph::edge_property_t<decltype(cur_graph_view), bool> edge_value_output_p_q(handle,
                                                                                 cur_graph_view);

      // rename the below to edges_with_triangles
      edges_with_triangles.insert(edgelist_srcs.begin(),
                       edgelist_srcs.begin() + last_edge_with_triangles_idx,
                       edgelist_dsts.begin());                       

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
        edge_value_output_p_q.mutable_view(),
        false);

      cur_graph_view.attach_edge_mask(edge_value_output_p_q.view());

      // resize the 'edgelist_srcs' and 'edgelsit_dst'
      edgelist_srcs.resize(last_edge_with_triangles_idx, handle.get_stream());
      edgelist_dsts.resize(last_edge_with_triangles_idx, handle.get_stream());
      num_triangles.resize(last_edge_with_triangles_idx, handle.get_stream());

      num_edges = edgelist_srcs.size();

      //num_invalid_edges = 0; //****************** debugging purposes
    }

    if (num_invalid_edges == num_edges) {
      // return empty graph view
      std::optional<graph_view_t<vertex_t, edge_t, false, multi_gpu>> empty_graph_view{std::nullopt};
      // FIXME: To be updated
      // return empty_graph_view;
    }
    else{
      // FIXME: To be updated
      // return cur_graph_view;
    }
  
    
  
  }
}

}  // namespace cugraph
