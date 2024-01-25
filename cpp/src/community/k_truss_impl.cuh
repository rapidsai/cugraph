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
#include <prims/extract_transform_e.cuh>
#include <prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh>
#include <prims/update_edge_src_dst_property.cuh>
#include <prims/reduce_op.cuh>
#include <prims/edge_bucket.cuh>
#include <prims/transform_e.cuh>

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
  __device__ bool operator()(edge_t core_number) const {return core_number >= k+1; }
};

template <typename vertex_t>
struct extract_k_plus_one_core_t {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(
    vertex_t src, vertex_t dst, bool src_in_k_plus_one_core, bool dst_in_k_plus_one_core, thrust::nullopt_t) const
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

  VertexPairIterator vertex_pairs_begin;

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(thrust::seq, intersection_offsets.begin()+1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin()+1, itr);
    thrust::tuple<vertex_t, vertex_t> pair = *(vertex_pairs_begin + idx);
   return pair;
  }
};


template <typename vertex_t, typename edge_t, typename VertexPairIterator>
struct extract_p_r {
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};

  VertexPairIterator vertex_pairs_begin;


  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(thrust::seq, intersection_offsets.begin()+1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin()+1, itr);
    auto pair = thrust::make_tuple(thrust::get<0>(*(vertex_pairs_begin + idx)), intersection_indices[i]);

    return pair;
  }
};



template <typename vertex_t, typename edge_t, typename VertexPairIterator>
struct extract_q_r {
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  VertexPairIterator vertex_pairs_begin;

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(thrust::seq, intersection_offsets.begin()+1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin()+1, itr);
    auto pair = thrust::make_tuple(thrust::get<1>(*(vertex_pairs_begin + idx)), intersection_indices[i]);

    return pair;
  }
};


template <typename vertex_t, typename edge_t, typename VertexPairIterator>
struct generate_pr {
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};

  VertexPairIterator vertex_pairs_begin;

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(thrust::seq, intersection_offsets.begin()+1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin()+1, itr);
    auto pair = thrust::make_tuple(thrust::get<0>(*(vertex_pairs_begin + idx)), intersection_indices[i]);

    return pair;
  }
};


template <typename vertex_t, typename edge_t, typename VertexPairIterator>
struct generate_qr {
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};

  VertexPairIterator vertex_pairs_begin;

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(thrust::seq, intersection_offsets.begin()+1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin()+1, itr);
    auto pair = thrust::make_tuple(thrust::get<1>(*(vertex_pairs_begin + idx)), intersection_indices[i]);

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


} // namespace

template <typename vertex_t, typename edge_t, bool multi_gpu>
void k_truss(raft::handle_t const& handle,
            graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
            edge_t k,
            bool do_expensive_check)
{
  using weight_t = float;  // dummy

  // 1. Check input arguments.

  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  CUGRAPH_EXPECTS(
    graph_view.is_symmetric(),
    "Invalid input arguments: K-truss currently supports undirected graphs only.");
  CUGRAPH_EXPECTS(
    !graph_view.is_multigraph(),
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

  // 3. Find (k+1)-core and exclude edges that do not belong to (k+1)-core (FIXME: better mask-out once we
  // add masking support).

  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;
    auto vertex_partition_range_lasts =
      renumber_map
        ? std::make_optional<std::vector<vertex_t>>(cur_graph_view.vertex_partition_range_lasts())
        : std::nullopt;

    rmm::device_uvector<edge_t> core_numbers(cur_graph_view.local_vertex_partition_range_size(),
                                             handle.get_stream());
    core_number(
      handle, cur_graph_view, core_numbers.data(), k_core_degree_type_t::OUT, size_t{k+1}, size_t{k+1});

    edge_src_property_t<decltype(cur_graph_view), bool> edge_src_in_k_plus_one_cores(handle,
                                                                              cur_graph_view);
    edge_dst_property_t<decltype(cur_graph_view), bool> edge_dst_in_k_plus_one_cores(handle,
                                                                              cur_graph_view);
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
  
  // 5. Decompress the resulting graph to an edge list and ind intersection of edge endpoints
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

    auto vertex_pairs_begin = thrust::make_zip_iterator(
      edgelist_srcs.begin(), edgelist_dsts.begin());
    
    thrust::sort(handle.get_thrust_policy(),
                 vertex_pairs_begin,
                 vertex_pairs_begin + edgelist_srcs.size());
    
    RAFT_CUDA_TRY(cudaDeviceSynchronize()); // FIXME: Only for debugging purposes, remove
    raft::print_device_vector("src ", edgelist_srcs.data(), edgelist_srcs.size(), std::cout);
    raft::print_device_vector("dst ", edgelist_dsts.data(), edgelist_dsts.size(), std::cout);

    size_t num_vertex_pairs = edgelist_srcs.size();
    auto out_degrees = cur_graph_view.compute_out_degrees(handle);

    rmm::device_uvector<edge_t> r_nbr_intersection_property_values0(size_t{0}, handle.get_stream()); // FIXME: remove as it is not used
    rmm::device_uvector<edge_t> r_nbr_intersection_property_values1(size_t{0}, handle.get_stream()); // FIXME: remove as it is not used

    // FIXME: Initially each edge should have an edge property of 0
    auto[intersection_offsets, intersection_indices] = 
      detail::nbr_intersection(handle,
                               cur_graph_view,
                               cugraph::edge_dummy_property_t{}.view(),
                               vertex_pairs_begin,
                               vertex_pairs_begin + num_vertex_pairs,
                               std::array<bool, 2>{true, true},
                               do_expensive_check);
    
    RAFT_CUDA_TRY(cudaDeviceSynchronize()); // FIXME: Only for debugging purposes, remove

    auto vertex_pair_buffer = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          num_vertex_pairs, handle.get_stream());

    
    // stores all the pairs (p, q), (p, r) and (q, r)
    auto vertex_pair_buffer_tmp = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          intersection_indices.size() * 3, handle.get_stream());

    // FIXME: optmize this part to not have to iterate over all the edges again
    // tabulate with the size of intersection_indices, and call binary search on intersection_offsets
    // to get (p, q).
    thrust::tabulate(handle.get_thrust_policy(),
                     get_dataframe_buffer_begin(vertex_pair_buffer_tmp),
                     get_dataframe_buffer_begin(vertex_pair_buffer_tmp) + intersection_indices.size(),
                     extract_p_q<vertex_t, edge_t, decltype(vertex_pairs_begin)>{
                     raft::device_span<size_t const>(
                       intersection_offsets.data(), intersection_offsets.size()),
                     raft::device_span<vertex_t const>(
                       intersection_indices.data(), intersection_indices.size()),
                     vertex_pairs_begin
                    });

   // tabulate with the size of intersection_indices, and call binary search on intersection_offsets
   // to get (p, r).
   thrust::tabulate(handle.get_thrust_policy(),
                    get_dataframe_buffer_begin(vertex_pair_buffer_tmp) + intersection_indices.size(),
                    get_dataframe_buffer_begin(vertex_pair_buffer_tmp) + (2 * intersection_indices.size()),
                    extract_p_r<vertex_t, edge_t, decltype(vertex_pairs_begin)>{
                    raft::device_span<size_t const>(
                      intersection_offsets.data(), intersection_offsets.size()),
                    raft::device_span<vertex_t const>(
                      intersection_indices.data(), intersection_indices.size()),
                    vertex_pairs_begin
                    });
   
   // tabulate with the size of intersection_indices, and call binary search on intersection_offsets
   // to get (q, r).
   thrust::tabulate(handle.get_thrust_policy(),
                    get_dataframe_buffer_begin(vertex_pair_buffer_tmp) + (2 * intersection_indices.size()),
                    get_dataframe_buffer_begin(vertex_pair_buffer_tmp) + (3 * intersection_indices.size()),
                    extract_q_r<vertex_t, edge_t, decltype(vertex_pairs_begin)>{
                    raft::device_span<size_t const>(
                      intersection_offsets.data(), intersection_offsets.size()),
                    raft::device_span<vertex_t const>(
                      intersection_indices.data(), intersection_indices.size()),
                    vertex_pairs_begin
                    });

    thrust::sort(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(vertex_pair_buffer_tmp),
                 get_dataframe_buffer_end(vertex_pair_buffer_tmp));
    
    rmm::device_uvector<vertex_t> num_triangles_tmp(3 * intersection_indices.size(), handle.get_stream());

    thrust::fill(handle.get_thrust_policy(), num_triangles_tmp.begin(), num_triangles_tmp.end(), size_t{1});

    rmm::device_uvector<vertex_t> num_triangles(num_vertex_pairs, handle.get_stream());

    // FIXME: Does a reduce by key preserve the sorting? I believe 'YES"
    thrust::reduce_by_key(handle.get_thrust_policy(),
                          get_dataframe_buffer_begin(vertex_pair_buffer_tmp),
                          get_dataframe_buffer_end(vertex_pair_buffer_tmp),
                          num_triangles_tmp.begin(),
                          get_dataframe_buffer_begin(vertex_pair_buffer),
                          num_triangles.begin(),
                          thrust::equal_to<thrust::tuple<vertex_t, vertex_t>>{});


    // Note: ensure 'edge_list' and 'cur_graph_view' have the same transpose flag
    // one of the void
    cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true>edge_list(handle);

    edge_list.insert(std::get<0>(vertex_pair_buffer).begin(),
                     std::get<0>(vertex_pair_buffer).end(),
                     // num triangle here
                     std::get<1>(vertex_pair_buffer).begin());

    cugraph::edge_property_t<decltype(cur_graph_view), vertex_t> edge_value_output(handle,
                                                                                   cur_graph_view);
    thrust::sort(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(vertex_pair_buffer),
                 get_dataframe_buffer_end(vertex_pair_buffer),
                 thrust::less<thrust::tuple<vertex_t, vertex_t>>{}); // FIXME: Isn't this sorting redunddant?

    // FIXME: shuffle before sorting and reducing - Bring this part up
    if constexpr (multi_gpu) {
      thrust::copy(handle.get_thrust_policy(),
                std::get<0>(vertex_pair_buffer).data(),
                std::get<0>(vertex_pair_buffer).data() + std::get<0>(vertex_pair_buffer).size(),
                edgelist_srcs.begin());

      thrust::copy(handle.get_thrust_policy(),
                std::get<1>(vertex_pair_buffer).data(),
                std::get<1>(vertex_pair_buffer).data() + std::get<1>(vertex_pair_buffer).size(),
                edgelist_dsts.begin());
      
      edgelist_srcs.resize(std::get<0>(vertex_pair_buffer).size(), handle.get_stream());
      edgelist_dsts.resize(std::get<0>(vertex_pair_buffer).size(), handle.get_stream());

      std::tie(edgelist_srcs, edgelist_dsts, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                      edge_t,
                                                                                      weight_t,
                                                                                      int32_t>(
          handle, std::move(edgelist_srcs), std::move(edgelist_dsts), std::nullopt, std::nullopt, std::nullopt);
    }

    // Run thrust::stable_partition to place the edges with triangle counts smaller than K-2 at the end
    // zip iterator for the pair
  
    auto edges_to_num_triangles = thrust::make_zip_iterator(
      get_dataframe_buffer_begin(vertex_pair_buffer), num_triangles.begin()); // FIXME: Might want to rezip it after resizing the 'vertex_pair_buffer'
    
    // 'invalid_edge_first' marks the beginning of the edges to be removed
    auto invalid_edge_first = thrust::stable_partition(
            handle.get_thrust_policy(),
            edges_to_num_triangles,
            edges_to_num_triangles + num_triangles.size(),
            [k]__device__(auto e){
              auto num_triangles = thrust::get<1>(e);
              auto is_in_k_truss = num_triangles == 2;
              return num_triangles > k; // FIXME k-2

            }
            );

    size_t num_invalid_edges{0};
    num_invalid_edges =
      static_cast<size_t>(thrust::distance(invalid_edge_first, edges_to_num_triangles + num_triangles.size()));
    

  
    if (num_invalid_edges != 0) { // unroll and remove/mask edges

      // case 2: unroll (q, r)
  
      auto incoming_vertex_pairs_begin = thrust::make_zip_iterator(
        edgelist_dsts.begin(), edgelist_srcs.begin());
      
      // Sort the 'incoming_vertex_pairs_begin'
      thrust::sort(handle.get_thrust_policy(),
                   incoming_vertex_pairs_begin,
                   incoming_vertex_pairs_begin + edgelist_srcs.size()); // FIXME: No need to partition
      
      // For each (q, r) edge to unroll, find the incoming edges to 'r' let's say from 'p' and
      // create the pair (p, q)

      rmm::device_uvector<int> prefix_sum(edgelist_srcs.size() + 1, handle.get_stream());
      thrust::tabulate(handle.get_thrust_policy(),
                       prefix_sum.begin(),
                       prefix_sum.end(),
                      [
                       invalid_first = thrust::get<0>(edges_to_num_triangles.get_iterator_tuple()),
                       incoming_vertex_pairs_begin = incoming_vertex_pairs_begin,
                       num_edges = edgelist_srcs.size()
                      ]
                      __device__(auto idx){
                      auto src =  thrust::get<0>(*(invalid_first + idx));
                      auto dst =  thrust::get<1>(*(invalid_first + idx));
                      auto dst_array_begin = thrust::get<0>(incoming_vertex_pairs_begin.get_iterator_tuple());
                      auto dst_array_end = thrust::get<0>(incoming_vertex_pairs_begin.get_iterator_tuple()) + num_edges;
                      auto itr_lower = thrust::lower_bound(thrust::seq, dst_array_begin, dst_array_end, dst);
                      auto idx_lower = thrust::distance(dst_array_begin, itr_lower); // FIXME: remove self loops
                      auto itr_upper = thrust::upper_bound(thrust::seq, dst_array_begin, dst_array_end, dst);
                      auto idx_upper = thrust::distance(dst_array_begin, itr_upper);
                      auto dist = thrust::distance(itr_lower, itr_upper);
                      return dist;
                      } );

      thrust::exclusive_scan(handle.get_thrust_policy(), prefix_sum.begin(), prefix_sum.end(), prefix_sum.begin());

      raft::print_device_vector("p_sum", prefix_sum.data(), prefix_sum.size(), std::cout);

      auto vertex_pair_buffer_p_q = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          prefix_sum.back_element(handle.get_stream()), handle.get_stream());

      auto vertex_pair_buffer_p_r = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          prefix_sum.back_element(handle.get_stream()), handle.get_stream());

      rmm::device_uvector<vertex_t> indices(edgelist_srcs.size(), handle.get_stream());
      thrust::tabulate(handle.get_thrust_policy(), indices.begin(), indices.end(), thrust::identity<vertex_t>());

      thrust::for_each(
        handle.get_thrust_policy(),
        indices.begin(),
        indices.end(),
          [invalid_first_dst = thrust::get<1>(thrust::get<0>(edges_to_num_triangles.get_iterator_tuple()).get_iterator_tuple()),
          invalid_first_src = thrust::get<0>(thrust::get<0>(edges_to_num_triangles.get_iterator_tuple()).get_iterator_tuple()),
          prefix_sum = prefix_sum.data(),
          incoming_vertex_pairs_begin = incoming_vertex_pairs_begin,
          vertex_pair_buffer_p_q = get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
          vertex_pair_buffer_p_r = get_dataframe_buffer_begin(vertex_pair_buffer_p_r),
          num_edges = edgelist_srcs.size()]
        __device__(auto idx) {
          auto src = invalid_first_src[idx];
          auto dst = invalid_first_dst[idx];
          auto dst_array_begin = invalid_first_dst;
          auto dst_array_end   = invalid_first_dst + num_edges;

          auto itr_lower = thrust::lower_bound(thrust::seq, dst_array_begin, dst_array_end, dst);
          auto idx_lower = thrust::distance(dst_array_begin, itr_lower); // Need a binary search to find the begining of the range

          auto dist = prefix_sum[idx + 1] - prefix_sum[idx];
        
          thrust::tabulate(thrust::seq,
                          vertex_pair_buffer_p_q + prefix_sum[idx],
                          vertex_pair_buffer_p_q + prefix_sum[idx] + dist,
                          [incoming_vertex_pairs_begin = incoming_vertex_pairs_begin,
                            dst                         = dst,
                            src                         = src,
                            idx_lower                   = idx_lower](auto idx_in_segment) {
                        
                            return thrust::make_tuple(
                              thrust::get<1>(*(incoming_vertex_pairs_begin + idx_lower +idx_in_segment)), src);
                          });
          
          thrust::tabulate(thrust::seq,
                          vertex_pair_buffer_p_r + prefix_sum[idx],
                          vertex_pair_buffer_p_r + prefix_sum[idx] + dist,
                          [incoming_vertex_pairs_begin = incoming_vertex_pairs_begin,
                            dst                         = dst,
                            src                         = src,
                            idx_lower                   = idx_lower](auto idx_in_segment) {
          
                            return thrust::make_tuple(
                              thrust::get<1>(*(incoming_vertex_pairs_begin + idx_lower +idx_in_segment)), dst);
                          });
        
        });

        // sort the edges
        thrust::sort(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(vertex_pair_buffer_p_q),
                 get_dataframe_buffer_end(vertex_pair_buffer_p_q),
                 thrust::less<thrust::tuple<vertex_t, vertex_t>>{}); // FIXME: Check if you need to sort this

        auto edge_exists = cur_graph_view.has_edge(
          handle,
          raft::device_span<vertex_t const>(std::get<0>(vertex_pair_buffer_p_q).data(), std::get<0>(vertex_pair_buffer_p_q).size()),
          raft::device_span<vertex_t const>(std::get<1>(vertex_pair_buffer_p_q).data(), std::get<1>(vertex_pair_buffer_p_q).size()));

        auto edge_to_edge_exists = thrust::make_zip_iterator(
          get_dataframe_buffer_begin(vertex_pair_buffer_p_q), edge_exists.begin()); // FIXME: Might want to rezip it after resizing the 'vertex_pair_buffer'
        
        auto has_edge_last = thrust::stable_partition(
                handle.get_thrust_policy(),
                edge_to_edge_exists,
                edge_to_edge_exists + edge_exists.size(),
                []__device__(auto e){
                  auto edge_exists = thrust::get<1>(e);
                  return edge_exists;

                }
                );
      
      // case 1. For the (p, q), find intersection 'r' to create (p, r, -1) and (q, r, -1) 
      auto[intersection_offsets, intersection_indices] = 
        detail::nbr_intersection(handle,
                                cur_graph_view,
                                cugraph::edge_dummy_property_t{}.view(),
                                thrust::get<0>(invalid_edge_first.get_iterator_tuple()),
                                thrust::get<0>(edges_to_num_triangles.get_iterator_tuple()) + num_triangles.size(),
                                std::array<bool, 2>{true, true},
                                do_expensive_check);

      RAFT_CUDA_TRY(cudaDeviceSynchronize()); // FIXME: Only for debugging purposes, remove

      auto prev_size = size_dataframe_buffer(vertex_pair_buffer);
      auto new_size = prev_size + intersection_indices.size();
      resize_dataframe_buffer(vertex_pair_buffer, prev_size + intersection_indices.size(), handle.get_stream());
      // resize num_triangles
      num_triangles.resize(new_size, handle.get_stream());
      // Fill the newly allocated space with -1
      thrust::fill(handle.get_thrust_policy(), num_triangles.begin() + prev_size, num_triangles.end(), size_t{-1});

      edges_to_num_triangles = thrust::make_zip_iterator(
        get_dataframe_buffer_begin(vertex_pair_buffer), num_triangles.begin()); // FIXME: Might want to rezip it after resizing the 'vertex_pair_buffer'

      // Unroll (p, q)

      // 1) generating (p, r)
      thrust::tabulate(handle.get_thrust_policy(),
                      get_dataframe_buffer_begin(vertex_pair_buffer) + prev_size,
                      get_dataframe_buffer_begin(vertex_pair_buffer) + new_size,
                      generate_pr<vertex_t, edge_t, decltype(vertex_pairs_begin)>{ //FIXME: it is supposed to be of the type of the last argument
                      raft::device_span<size_t const>(
                        intersection_offsets.data(), intersection_offsets.size()),
                      raft::device_span<vertex_t const>(
                        intersection_indices.data(), intersection_indices.size()),
                      vertex_pairs_begin//thrust::get<0>(edges_to_num_triangles.get_iterator_tuple())
                      });

      

      prev_size = size_dataframe_buffer(vertex_pair_buffer);
      new_size = prev_size + intersection_indices.size();
      resize_dataframe_buffer(vertex_pair_buffer, prev_size + intersection_indices.size(), handle.get_stream());
      // resize num_triangles
      num_triangles.resize(new_size, handle.get_stream());
      // Fill the newly allocated space with -1
      thrust::fill(handle.get_thrust_policy(), num_triangles.begin() + prev_size, num_triangles.end(), size_t{-1});

      edges_to_num_triangles = thrust::make_zip_iterator(
        get_dataframe_buffer_begin(vertex_pair_buffer), num_triangles.begin()); // FIXME: Might want to rezip it after resizing the 'vertex_pair_buffer'

    }
    

  }

}

} // namespace cugraph
