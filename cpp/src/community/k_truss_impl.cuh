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
  edge_t k;
  __device__ bool operator()(edge_t core_number) const {return core_number >= k; }
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
    printf("\n i = %d", i);
    auto itr = thrust::upper_bound(thrust::seq, intersection_offsets.begin()+1, intersection_offsets.end(), i);

    auto idx = thrust::distance(intersection_offsets.begin()+1, itr);
    printf("\nthe idx = %d", idx);

    printf("\n major = %d and minor = %d", thrust::get<0>(*(vertex_pairs_begin + idx)), thrust::get<1>(*(vertex_pairs_begin + idx)));
    printf("\nintersection = %d\n", intersection_offsets[i+1] - intersection_offsets[i]);

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

    printf("\n ** (p, q) major = %d and minor = %d", thrust::get<0>(*(vertex_pairs_begin + idx)), intersection_indices[i]);
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
    printf("\n i = %d", i);
    auto itr = thrust::upper_bound(thrust::seq, intersection_offsets.begin()+1, intersection_offsets.end(), i);

    auto idx = thrust::distance(intersection_offsets.begin()+1, itr);

    printf("\n ** (p, q) major = %d and minor = %d", thrust::get<1>(*(vertex_pairs_begin + idx)), intersection_indices[i]);

   auto pair = thrust::make_tuple(thrust::get<1>(*(vertex_pairs_begin + idx)), intersection_indices[i]);

   return pair;
  }
};


template <typename vertex_t, typename edge_t>
struct triangle_counts_less_t {
  raft::device_span<vertex_t const> num_triangles{};
  edge_t k;

};

template <typename vertex_t, typename edge_t, typename VertexPairBuffer, typename Iterator>
struct find_intersection {
  size_t vertex_pair_buffer_size;
  size_t num_invalid_edges;
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};

  VertexPairBuffer vertex_pair_buffer;
  Iterator invalid_edge_first;

  __device__ size_t operator()(edge_t i) const
  {
    printf("\n ***** i = %d\n", i);
    printf("\n src = %d, dst = %d\n", thrust::get<0>(*(vertex_pair_buffer+i)), thrust::get<1>(*(vertex_pair_buffer+i)));
    printf("\nthe vertex pair buffer size = %d\n", vertex_pair_buffer_size);
    //auto x = thrust::get<1>(*invalid_edge_first);
    //printf("\nx = %d\n", x);
    // Run binary search
    //if (i < vertex_pair_buffer_size) {
    if (i < num_invalid_edges) {
      auto pair = thrust::get<0>(*(invalid_edge_first + i)); // FIXME only increment if it is less than 'invalid_edge_first.size()'
      printf("\nsrc_pair = %d, dst_pair = %d\n", thrust::get<0>(pair), thrust::get<1>(pair));
    
      auto itr = thrust::upper_bound(thrust::seq,
                      vertex_pair_buffer,
                      vertex_pair_buffer + vertex_pair_buffer_size,
                      //pair,
                      thrust::make_tuple(thrust::get<0>(pair), thrust::get<1>(pair)),
                      thrust::less<thrust::tuple<vertex_t, vertex_t>>{});
      
      printf("\n itr = %d\n", itr);
      auto idx = thrust::distance(vertex_pair_buffer + 1, itr);
      printf("\n idx = %d \n", idx);
      
      return 1;
    }
    else {
      return 0;
    }


   return 0;
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
    "Invalid input arguments: triangle_count currently supports undirected graphs only.");
  CUGRAPH_EXPECTS(
    !graph_view.is_multigraph(),
    "Invalid input arguments: triangle_count currently does not support multi-graphs.");
  

  // 2. Exclude self-loops (FIXME: better mask-out once we add masking support).

  std::optional<graph_t<vertex_t, edge_t, false, multi_gpu>> modified_graph{std::nullopt};
  std::optional<graph_view_t<vertex_t, edge_t, false, multi_gpu>> modified_graph_view{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t>> edge_ids{std::nullopt};
  // FIXME: Maybe should not be optional
  std::optional<rmm::device_uvector<edge_t>> wgts{std::nullopt};
  std::optional<rmm::device_uvector<edge_t>> wgts_{std::nullopt};

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

    rmm::device_uvector<edge_t> core_numbers(cur_graph_view.number_of_vertices(),
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

  rmm::device_uvector<edge_t> cur_graph_counts(size_t{0}, handle.get_stream());
  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;
    cur_graph_counts.resize(cur_graph_view.local_vertex_partition_range_size(),
                            handle.get_stream());

    rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<edge_t>> edgelist_prop {std::nullopt};


    std::tie(edgelist_srcs, edgelist_dsts, std::ignore, std::ignore) = decompress_to_edgelist(
        handle,
        cur_graph_view,
        std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
        std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<raft::device_span<vertex_t const>>(std::nullopt));

    auto vertex_pairs_begin = thrust::make_zip_iterator(
      edgelist_srcs.begin(), edgelist_dsts.begin());
    
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector("src ", edgelist_srcs.data(), edgelist_srcs.size(), std::cout);
    raft::print_device_vector("dst ", edgelist_dsts.data(), edgelist_dsts.size(), std::cout);

    size_t num_vertex_pairs = edgelist_srcs.size();
    auto out_degrees = cur_graph_view.compute_out_degrees(handle);

    rmm::device_uvector<size_t> intersection_offsets(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> intersection_indices(size_t{0}, handle.get_stream());
    rmm::device_uvector<edge_t> r_nbr_intersection_property_values0(size_t{0}, handle.get_stream());
    rmm::device_uvector<edge_t> r_nbr_intersection_property_values1(size_t{0}, handle.get_stream());

    // FIXME: Initially each edge should have an edge property of 0
    std::tie(intersection_offsets, intersection_indices) = 
      detail::nbr_intersection(handle,
                               cur_graph_view,
                               cugraph::edge_dummy_property_t{}.view(),
                               vertex_pairs_begin,
                               vertex_pairs_begin + num_vertex_pairs,
                               std::array<bool, 2>{true, true},
                               do_expensive_check);
    
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector("intersection_offsets ", intersection_offsets.data(), intersection_offsets.size(), std::cout);
    raft::print_device_vector("intersection_indices ", intersection_indices.data(), intersection_indices.size(), std::cout);

    auto vertex_pair_buffer = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          num_vertex_pairs, handle.get_stream());

    
    // stores all the pairs (p, q), (p, r) and (q, r)
    auto vertex_pair_buffer_ = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          intersection_indices.size() * 3, handle.get_stream());

    // FIXME: optmize this part to not have to iterate over all the edges again
    // tabulate with the size of intersection_indices, and call binary search on intersection_offsets
    // to get (p, q).
    thrust::tabulate(handle.get_thrust_policy(),
                     get_dataframe_buffer_begin(vertex_pair_buffer_),
                     get_dataframe_buffer_begin(vertex_pair_buffer_) + intersection_indices.size(),
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
                    get_dataframe_buffer_begin(vertex_pair_buffer_) + intersection_indices.size(),
                    get_dataframe_buffer_begin(vertex_pair_buffer_) + (2 * intersection_indices.size()),
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
                    get_dataframe_buffer_begin(vertex_pair_buffer_) + (2 * intersection_indices.size()),
                    get_dataframe_buffer_begin(vertex_pair_buffer_) + (3 * intersection_indices.size()),
                    extract_q_r<vertex_t, edge_t, decltype(vertex_pairs_begin)>{
                    raft::device_span<size_t const>(
                      intersection_offsets.data(), intersection_offsets.size()),
                    raft::device_span<vertex_t const>(
                      intersection_indices.data(), intersection_indices.size()),
                    vertex_pairs_begin
                    });
   
    printf("\nbefore sorting\n");
    raft::print_device_vector("src", std::get<0>(vertex_pair_buffer_).data(), std::get<0>(vertex_pair_buffer_).size(), std::cout);
    raft::print_device_vector("dst", std::get<1>(vertex_pair_buffer_).data(), std::get<1>(vertex_pair_buffer_).size(), std::cout);


    thrust::sort(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(vertex_pair_buffer_),
                 get_dataframe_buffer_end(vertex_pair_buffer_));
    
    
    rmm::device_uvector<vertex_t> num_triangles_(3 * intersection_indices.size(), handle.get_stream());

    thrust::fill(handle.get_thrust_policy(), num_triangles_.begin(), num_triangles_.end(), size_t{1});

    rmm::device_uvector<vertex_t> num_triangles(num_vertex_pairs, handle.get_stream());
  
    thrust::reduce_by_key(handle.get_thrust_policy(),
                          get_dataframe_buffer_begin(vertex_pair_buffer_),
                          get_dataframe_buffer_end(vertex_pair_buffer_),
                          num_triangles_.begin(),
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
                 thrust::less<thrust::tuple<vertex_t, vertex_t>>{});
    raft::print_device_vector("sorted - src", std::get<0>(vertex_pair_buffer).data(), std::get<0>(vertex_pair_buffer).size(), std::cout);
    raft::print_device_vector("sorted - dst", std::get<1>(vertex_pair_buffer).data(), std::get<1>(vertex_pair_buffer).size(), std::cout);
    raft::print_device_vector("new num triangles", num_triangles.data(), num_triangles.size(), std::cout);
    handle.sync_stream();
    cugraph::transform_e(
      handle,
      cur_graph_view,
      edge_list, // assigning th e number of triangles as an edge property
      cugraph::edge_src_dummy_property_t{}.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      cugraph::edge_dummy_property_t{}.view(),
      [num_triangles = num_triangles.data(),
        vertex_pair_buffer = get_dataframe_buffer_begin(vertex_pair_buffer),
        size=get_dataframe_buffer_end(vertex_pair_buffer) - get_dataframe_buffer_begin(vertex_pair_buffer)] // FIXME: use 'size_dataframe_buffer(vertex_pair_buffer)'
        // FIXME: Maybe: the last 'thrust::nullopt_t' for the edge prop containing the number of triangles.
        // Take a look at the 'vertex_bucket_t' for context
        __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {
        /*
        for ( int i =0 ; i < size; i++){
            //printf("\n i = %d major = %d and minor = %d", i, thrust::get<0>(*(vertex_pair_buffer+i)), thrust::get<1>(*(vertex_pair_buffer+i)));
            printf("hello");
        }
        */
  
    

        
        auto x = thrust::make_tuple(thrust::get<0>(*(vertex_pair_buffer)), thrust::get<1>(*(vertex_pair_buffer)));
        printf("\nsrc = %d , dst = %d\n", src, dst);
        //printf("\ny = %d\n", vertex_pair_buffer);

        auto it = thrust::lower_bound(thrust::seq,
                    vertex_pair_buffer,
                    vertex_pair_buffer + size,
                    thrust::make_tuple(src, dst));
        printf("\ntransform_e - itr = %d\n", it);
        auto idx = thrust::distance(vertex_pair_buffer, it);
        printf("\nthe distance = %d\n", idx);
        
        /*
        auto it_ = thrust::lower_bound(thrust::seq,
                    vertex_pair_buffer,
                    vertex_pair_buffer + size,
                    //thrust::make_tuple(src, dst),
                    thrust::make_tuple(1, 3),
                    thrust::less<thrust::tuple<vertex_t, vertex_t>>{});
        printf("\ntransform_e - itr____ = %d\n", it_);
        auto idx_ = thrust::distance(vertex_pair_buffer, it_);
        printf("\nthe distance = %d\n", idx_);
        */
        
        //printf("\nitr = %d\n", *it);
        return num_triangles[idx];
      },
      edge_value_output.mutable_view(),
      false);


      /*
      rmm::device_uvector<vertex_t> intersection_size_(intersection_offsets.size() - 1, handle.get_stream()); // take the worst case where all edges are invalidated
      // FIXME: might not be necessary to fill it with zeros
      thrust::fill(handle.get_thrust_policy(), intersection_size_.begin(), intersection_size_.end(), size_t{0});
      printf("\n*******dummy******\n");
      thrust::tabulate(handle.get_thrust_policy(),
                       intersection_size_.begin(),
                       intersection_size_.end(),
                       [vertex_pair_buffer_size=size_dataframe_buffer(vertex_pair_buffer),
                        intersection_offsets = raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
                        intersection_indices = raft::device_span<vertex_t const>(intersection_indices.data(), intersection_indices.size()),
                        vertex_pair_buffer = get_dataframe_buffer_begin(vertex_pair_buffer)
                        ]
                       __device__(edge_t i){
                          printf("\n ***** i = %d\n", i);
                          printf("\n src = %d, dst = %d\n", thrust::get<0>(*(vertex_pair_buffer+i)), thrust::get<1>(*(vertex_pair_buffer+i)));
                          printf("\nthe vertex pair buffer size = %d\n", vertex_pair_buffer_size);
                          auto num_invalid_edges = 6;
                          if (i < num_invalid_edges) {
                            //auto pair = thrust::get<0>(*(invalid_edge_first + i)); // FIXME only increment if it is less than 'invalid_edge_first.size()'
                            //printf("\nsrc_pair = %d, dst_pair = %d\n", thrust::get<0>(pair), thrust::get<1>(pair));
                          
                            auto itr = thrust::lower_bound(thrust::seq,
                                            vertex_pair_buffer,
                                            vertex_pair_buffer + vertex_pair_buffer_size,
                                            //pair,
                                            //thrust::make_tuple(thrust::get<0>(pair), thrust::get<1>(pair)),
                                            thrust::make_tuple(2, 3),
                                            thrust::less<thrust::tuple<vertex_t, vertex_t>>{});
                            
                            printf("\n itr = %d\n", itr);
                            auto idx = thrust::distance(vertex_pair_buffer, itr);
                            printf("\n idx = %d \n", idx);
                            
                            return 1;
                          }
                          else {
                            return 0;
                          }

                       });
      printf("\n*******Done dummy******\n");
      */









      
      
      auto value_firsts = edge_value_output.view().value_firsts();
      auto edge_counts  = edge_value_output.view().edge_counts();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      raft::print_device_vector(
        "num_triangles_:", value_firsts[0], edge_counts[0], std::cout);

    vertex_pairs_begin = thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer).begin(), std::get<1>(vertex_pair_buffer).begin()); // FIXME: not used below but used up
    printf("\nafter sorting\n");
    raft::print_device_vector("src", std::get<0>(vertex_pair_buffer).data(), std::get<0>(vertex_pair_buffer).size(), std::cout);
    raft::print_device_vector("dst", std::get<1>(vertex_pair_buffer).data(), std::get<1>(vertex_pair_buffer).size(), std::cout);

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

    // Run thrust::partition to place the edges with triangle counts smaller than K-2 at the end
    // zip iterator for the pair
    auto edges_to_num_triangles = thrust::make_zip_iterator(
      get_dataframe_buffer_begin(vertex_pair_buffer), num_triangles.begin());
    
    // FIXME: rename it to edge_to_remove_first
    auto invalid_edge_first = thrust::partition(
            handle.get_thrust_policy(),
            edges_to_num_triangles,
            edges_to_num_triangles + num_triangles.size(),
            [k]__device__(auto e){
              auto num_triangles = thrust::get<1>(e);
              auto is_in_k_truss = num_triangles == 2;
              printf("\n is in k_truss = %d\n", is_in_k_truss);
              return num_triangles > k; // FIXME k-2

            }
            );
    
    // FIXME: rename it num_edges_to_remove
    size_t num_invalid_edges{0};
    num_invalid_edges =
      static_cast<size_t>(thrust::distance(invalid_edge_first, edges_to_num_triangles + num_triangles.size()));
    
    printf("\nthe number of invalid edges = %d\n", num_invalid_edges);
    /*
      void resize_dataframe_buffer(BufferType& buffer,
                             size_t new_buffer_size,
                             rmm::cuda_stream_view stream_view)
    */
  
    if (num_invalid_edges != 0) { // unroll and remove/mask edges
      // Find (p, q, intersection of p&q's neighbor lists) 
      // before that, determine the interection size of each edge to be removed
      rmm::device_uvector<vertex_t> intersection_size(intersection_offsets.size() - 1, handle.get_stream()); // take the worst case where all edges are invalidated
      // FIXME: might not be necessary to fill it with zeros
      thrust::fill(handle.get_thrust_policy(), intersection_size.begin(), intersection_size.end(), size_t{0});
      /*
      thrust::tabulate(handle.get_thrust_policy(),
                       intersection_size.begin(),
                       intersection_size.end(),
                       find_intersection<vertex_t, edge_t, decltype(get_dataframe_buffer_begin(vertex_pair_buffer_)), decltype(invalid_edge_first)>{
                       //find_intersection<vertex_t, edge_t, decltype(vertex_pairs_begin), decltype(invalid_edge_first)>{
                       size_dataframe_buffer(vertex_pair_buffer), // FIXME: should we just do this inside the functor?
                       num_invalid_edges,
                       raft::device_span<size_t const>(
                         intersection_offsets.data(), intersection_offsets.size()),
                       raft::device_span<vertex_t const>(
                         intersection_indices.data(), intersection_indices.size()),
                       get_dataframe_buffer_begin(vertex_pair_buffer_), // FIXME: rename to vertex_pairs_buffer? maybe I should pass a pointer here 'get_dataframe_buffer_begin(vertex_pair_buffer)'
                       //vertex_pairs_begin,
                       invalid_edge_first
                       });
      */
      
      thrust::tabulate(handle.get_thrust_policy(),
                       intersection_size.begin(),
                       intersection_size.end(),
                       [vertex_pair_buffer_size=size_dataframe_buffer(vertex_pair_buffer),
                        num_invalid_edges=num_invalid_edges,
                        intersection_offsets = raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
                        intersection_indices = raft::device_span<vertex_t const>(intersection_indices.data(), intersection_indices.size()),
                        vertex_pair_buffer = get_dataframe_buffer_begin(vertex_pair_buffer),
                        invalid_edge_first = invalid_edge_first
                        ]
                       __device__(edge_t i){
                          
                          printf("\n ***** i = %d\n", i);
                          printf("\n src = %d, dst = %d\n", thrust::get<0>(*(vertex_pair_buffer+i)), thrust::get<1>(*(vertex_pair_buffer+i)));
                          printf("\nthe vertex pair buffer size = %d\n", vertex_pair_buffer_size);
                          if (i < num_invalid_edges) {
                            //auto pair = thrust::get<0>(*(invalid_edge_first + i)); // FIXME only increment if it is less than 'invalid_edge_first.size()'
                            vertex_t src = thrust::get<0>(thrust::get<0>(*(invalid_edge_first + i)));
                            vertex_t dst = thrust::get<1>(thrust::get<0>(*(invalid_edge_first + i)));
                            //printf("\nsrc_pair = %d, dst_pair = %d, pair = %d\n", thrust::get<0>(pair), thrust::get<1>(pair), pair);
                            printf("\nsrc_pair = %d, dst_pair = %d", src, dst);
                            
                            auto itr = thrust::lower_bound(thrust::seq,
                                            vertex_pair_buffer,
                                            vertex_pair_buffer + vertex_pair_buffer_size,
                                            //pair,
                                            //thrust::make_tuple(thrust::get<0>(pair), thrust::get<1>(pair)),
                                            //thrust::make_tuple(1, 3),
                                            thrust::make_tuple(src, dst),
                                            thrust::less<thrust::tuple<vertex_t, vertex_t>>{});
                            
                            printf("\n itr = %d\n", itr);
                            auto idx = thrust::distance(vertex_pair_buffer, itr);
                            printf("\n distance = %d \n", idx);
                            
                            return 1; // dummy
                          }
                          else {
                            return 0; //dummy
                          }

                       });
      

  


      


    }
    
    /*
    thrust::partition(
            handle.get_thrust_policy(),
            get_dataframe_buffer_begin(vertex_pair_buffer),
            get_dataframe_buffer_end(vertex_pair_buffer),
            [num_triangles = num_triangles.data(),
             vertex_pair_buffer = get_dataframe_buffer_begin(vertex_pair_buffer),
             //size=get_dataframe_buffer_end(vertex_pair_buffer) - get_dataframe_buffer_begin(vertex_pair_buffer)
             size = size_dataframe_buffer(vertex_pair_buffer)] //size_dataframe_buffer
            __device__(auto e){
              //printf("src = %d, dst = %d\n", std::get<0>(e), std::get<1>(e));
              auto[src, dst] = e;
              auto x = thrust::make_tuple(src, dst);
              printf("src_ = %d, dst_ = %d\n", thrust::get<0>(x), thrust::get<1>(x));
              //vertex_t src_ = thrust::get<0>(x);
              //vertex_t dst_ = thrust::get<1>(x);
              vertex_t dst_ = thrust::get<1>(e);
              //vertex_t dst__ = e;


              //auto x = thrust::make_tuple(src, dst);
              //printf("src = %d, dst = %d\n", src, dst);
              //printf("src_ = %d, dst_ = %d\n", thrust::get<0>(x), thrust::get<1>(x));
              //vertex_t src_ = thrust::get<0>(x);
              //printf("src___ = %d\n", src_);
              //vertex_t src_ = thrust::get<0>(x);
              //vertex_t dst_ = thrust::get<1>(x); // error: no suitable conversion function from "thrust::detail::cons<int &, thrust::null_type>" to "int32_t" exists



             // sort by key. key num of triangles and the values =vertex pairs. thrust::greater
             printf("the isze of the buffer = %d\n", size);
              auto it = thrust::lower_bound(thrust::seq,
                    vertex_pair_buffer,
                    vertex_pair_buffer + size,
                    e,
                    thrust::less<thrust::tuple<vertex_t, vertex_t>>{});

              
              auto idx = thrust::distance(vertex_pair_buffer, it);
              printf("\nthe distance = %d\n", idx);
              
              return 0;

            }
            );
    */       
  


  }

}

} // namespace cugraph
