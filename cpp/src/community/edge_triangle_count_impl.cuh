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

#include "detail/graph_partition_utils.cuh"
#include "prims/edge_bucket.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh"

// FIXME:::: Remove ************************************************************
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
// FIXME:::: Remove ************************************************************

#include <cugraph/graph_functions.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename EdgeIterator>
struct update_edges_p_r_q_r_num_triangles {
  size_t num_edges{};
  const edge_t edge_first_or_second{};
  size_t chunk_start{};
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  raft::device_span<edge_t> num_triangles{};

  EdgeIterator edge_first{};

  __device__ void operator()(size_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);
    if (edge_first_or_second == 0) {
      auto p_r_pair = thrust::make_tuple(thrust::get<0>(*(edge_first + chunk_start + idx)),
                                         intersection_indices[i]);

      // Find its position in 'edges'
      auto itr_p_r_p_q =
        thrust::lower_bound(thrust::seq, edge_first, edge_first + num_edges, p_r_pair);

      assert(*itr_p_r_p_q == p_r_pair);
      idx = thrust::distance(edge_first, itr_p_r_p_q);
    } else {
      auto p_r_pair = thrust::make_tuple(thrust::get<1>(*(edge_first + chunk_start + idx)),
                                         intersection_indices[i]);

      // Find its position in 'edges'
      auto itr_p_r_p_q =
        thrust::lower_bound(thrust::seq, edge_first, edge_first + num_edges, p_r_pair);
      assert(*itr_p_r_p_q == p_r_pair);
      idx = thrust::distance(edge_first, itr_p_r_p_q);
    }
    cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(num_triangles[idx]);
    auto r = atomic_counter.fetch_add(edge_t{1}, cuda::std::memory_order_relaxed);
  }
};

template <typename vertex_t, typename edge_t, typename EdgeIterator>
struct extract_p_r_q_r {
  size_t chunk_start{};
  size_t p_r_or_q_r{};
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  EdgeIterator edge_first;

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(thrust::seq, intersection_offsets.begin()+1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin()+1, itr);

    if (p_r_or_q_r == 0) {
      return thrust::make_tuple(thrust::get<0>(*(edge_first + chunk_start + idx)), intersection_indices[i]);
    } else {
      return thrust::make_tuple(thrust::get<1>(*(edge_first + chunk_start + idx)), intersection_indices[i]);
    }

  }
};


template <typename vertex_t, typename edge_t, typename EdgeIterator>
struct extract_q_r {
  size_t chunk_start{};
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  EdgeIterator edge_first;


  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(thrust::seq, intersection_offsets.begin()+1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin()+1, itr);
    auto pair = thrust::make_tuple(thrust::get<1>(*(edge_first + chunk_start + idx)), intersection_indices[i]);

    return pair;
  }
};


template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t>
edge_triangle_count_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view)
{
  using weight_t = float;
  rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
  std::tie(edgelist_srcs, edgelist_dsts, std::ignore, std::ignore) = decompress_to_edgelist(
    handle,
    graph_view,
    std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
    std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
    std::optional<raft::device_span<vertex_t const>>(std::nullopt));

  auto edge_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());

  thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + edgelist_srcs.size());

  size_t approx_edges_to_intersect_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 17);
  
  approx_edges_to_intersect_per_iteration = 4;

  auto num_chunks        = ((edgelist_srcs.size() % approx_edges_to_intersect_per_iteration) == 0)
                             ? (edgelist_srcs.size() / approx_edges_to_intersect_per_iteration)
                             : (edgelist_srcs.size() / approx_edges_to_intersect_per_iteration) + 1;
  
  // Note: host_scalar_all_reduce to get the max reduction
  // Note: edge src dst and delta -> shuffle those -> and update -> check this : shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning
  // Note: shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning in shuffle_wrapper
  size_t prev_chunk_size = 0;
  auto num_edges         = edgelist_srcs.size();
  rmm::device_uvector<edge_t> num_triangles(edgelist_srcs.size(), handle.get_stream());

  //auto my_rank = handle.get_comms().get_rank();
  if constexpr (multi_gpu) {
    num_chunks = host_scalar_allreduce(
          handle.get_comms(), num_chunks, raft::comms::op_t::MAX, handle.get_stream());
  }

  printf("\n initial edgelists, num_chunk = %d\n", num_chunks);

  raft::print_device_vector("edgelist_srcs", edgelist_srcs.data(), edgelist_srcs.size(), std::cout);
  raft::print_device_vector("edgelist_dsts", edgelist_dsts.data(), edgelist_dsts.size(), std::cout);

  // Need to ensure that the vector has its values initialized to 0 before incrementing
  thrust::fill(handle.get_thrust_policy(), num_triangles.begin(), num_triangles.end(), 0);

  for (size_t i = 0; i < num_chunks; ++i) {
    auto chunk_size = std::min(approx_edges_to_intersect_per_iteration, num_edges);
    printf("\niteration = %d, chunk_size = %d\n", i, chunk_size);
    num_edges -= chunk_size;
    // Perform 'nbr_intersection' in chunks to reduce peak memory.
    auto [intersection_offsets, intersection_indices] =
      detail::nbr_intersection(handle,
                               graph_view,
                               cugraph::edge_dummy_property_t{}.view(),
                               edge_first + prev_chunk_size,
                               edge_first + prev_chunk_size + chunk_size,
                               std::array<bool, 2>{true, true},
                               false /*FIXME: pass 'do_expensive_check' as argument*/);
    
    printf("\nchunk processed\n");
    raft::print_device_vector("edgelist_srcs", edgelist_srcs.data() + prev_chunk_size, chunk_size, std::cout);
    raft::print_device_vector("edgelist_dsts", edgelist_dsts.data() + prev_chunk_size, chunk_size, std::cout);
    raft::print_device_vector("offsets", intersection_offsets.data(), intersection_offsets.size(), std::cout);
    raft::print_device_vector("indices", intersection_indices.data(), intersection_indices.size(), std::cout);
    printf("\n");

    // Update the number of triangles of each (p, q) edges by looking at their intersection
    // size

    
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator<edge_t>(0),
      thrust::make_counting_iterator<edge_t>(chunk_size),
      [chunk_start          = prev_chunk_size,
       num_triangles        = raft::device_span<edge_t>(num_triangles.data(), num_triangles.size()),
       intersection_offsets = raft::device_span<size_t const>(
         intersection_offsets.data(), intersection_offsets.size())] __device__(auto i) {
        num_triangles[chunk_start + i] += (intersection_offsets[i + 1] - intersection_offsets[i]);
      });

    if constexpr (multi_gpu) {
      // stores all the pairs (p, r) and (q, r)
      auto vertex_pair_buffer_tmp = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
            intersection_indices.size() * 2, handle.get_stream()); // *2 for both (p, r) and (q, r)
      // So that you shuffle only once

      // tabulate with the size of intersection_indices, and call binary search on intersection_offsets
      // to get (p, r).
      thrust::tabulate(handle.get_thrust_policy(),
                        get_dataframe_buffer_begin(vertex_pair_buffer_tmp),
                        get_dataframe_buffer_begin(vertex_pair_buffer_tmp) + intersection_indices.size(),
                        extract_p_r_q_r<vertex_t, edge_t, decltype(edge_first)>{
                        prev_chunk_size,
                        0,
                        raft::device_span<size_t const>(
                          intersection_offsets.data(), intersection_offsets.size()),
                        raft::device_span<vertex_t const>(
                          intersection_indices.data(), intersection_indices.size()),
                        edge_first
                        });
      // FIXME: Consolidate both functions
      thrust::tabulate(handle.get_thrust_policy(),
                        get_dataframe_buffer_begin(vertex_pair_buffer_tmp) + intersection_indices.size(),
                        get_dataframe_buffer_begin(vertex_pair_buffer_tmp) + (2 * intersection_indices.size()),
                        extract_p_r_q_r<vertex_t, edge_t, decltype(edge_first)>{
                        prev_chunk_size,
                        1,
                        raft::device_span<size_t const>(
                          intersection_offsets.data(), intersection_offsets.size()),
                        raft::device_span<vertex_t const>(
                          intersection_indices.data(), intersection_indices.size()),
                        edge_first
                        });

      thrust::sort(handle.get_thrust_policy(),
                  get_dataframe_buffer_begin(vertex_pair_buffer_tmp),
                  get_dataframe_buffer_end(vertex_pair_buffer_tmp));
      
      printf("\np, r and q, r\n");
      raft::print_device_vector("edgelist_srcs", std::get<0>(vertex_pair_buffer_tmp).data(), std::get<0>(vertex_pair_buffer_tmp).size(), std::cout);
      raft::print_device_vector("edgelist_dsts", std::get<1>(vertex_pair_buffer_tmp).data(), std::get<1>(vertex_pair_buffer_tmp).size(), std::cout);
      
      rmm::device_uvector<edge_t> increase_count_tmp(2 * intersection_indices.size(), handle.get_stream());
      thrust::fill(handle.get_thrust_policy(), increase_count_tmp.begin(), increase_count_tmp.end(), size_t{1});

      auto count_p_r_q_r = thrust::unique_count(handle.get_thrust_policy(),
                                            get_dataframe_buffer_begin(vertex_pair_buffer_tmp),
                                            get_dataframe_buffer_end(vertex_pair_buffer_tmp));

      rmm::device_uvector<edge_t> increase_count(count_p_r_q_r, handle.get_stream());
      

      auto vertex_pair_buffer = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
            count_p_r_q_r, handle.get_stream());
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            get_dataframe_buffer_begin(vertex_pair_buffer_tmp),
                            get_dataframe_buffer_end(vertex_pair_buffer_tmp),
                            increase_count_tmp.begin(),
                            get_dataframe_buffer_begin(vertex_pair_buffer),
                            increase_count.begin(),
                            thrust::equal_to<thrust::tuple<vertex_t, vertex_t>>{});

      rmm::device_uvector<vertex_t> pair_srcs(0, handle.get_stream());
      rmm::device_uvector<vertex_t> pair_dsts(0, handle.get_stream());
      std::optional<rmm::device_uvector<edge_t>> pair_count{std::nullopt};

      std::optional<rmm::device_uvector<edge_t>> opt_increase_count =
          std::make_optional(rmm::device_uvector<edge_t>(increase_count.size(), handle.get_stream()));

      raft::copy<edge_t>((*opt_increase_count).begin(),
                          increase_count.begin(),
                          increase_count.size(),
                          handle.get_stream());

      // There are still multiple copies here but is it worth sorting and reducing again?
      std::tie(pair_srcs, pair_dsts, std::ignore, pair_count, std::ignore) = shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t, edge_t, weight_t, int32_t>(
        handle,
        std::move(std::get<0>(vertex_pair_buffer)),
        std::move(std::get<1>(vertex_pair_buffer)),
        std::nullopt,
        std::move(opt_increase_count),
        std::nullopt,
        graph_view.vertex_partition_range_lasts());
  
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(pair_srcs.size()),
        [num_edges = edgelist_srcs.size(),
         num_triangles = num_triangles.data(),
         pair_srcs = pair_srcs.data(),
         pair_dsts = pair_dsts.data(),
         pair_count = (*pair_count).data(),
         edge_first]
        __device__(auto idx) {
          auto src = pair_srcs[idx];
          auto dst = pair_dsts[idx];
          auto p_r_q_r_pair = thrust::make_tuple(src, dst);
      
          // Find its position in 'edges'
          auto itr_p_r_q_r =
            thrust::lower_bound(thrust::seq, edge_first, edge_first + num_edges, p_r_q_r_pair);
        
          assert(*itr_p_r_q_r == p_r_q_r_pair);
          auto idx_p_r_q_r = thrust::distance(edge_first, itr_p_r_q_r);

          cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(num_triangles[idx_p_r_q_r]);
          auto r = atomic_counter.fetch_add(pair_count[idx], cuda::std::memory_order_relaxed);

        }
      );

    } else {
    
      // Given intersection offsets and indices that are used to update the number of
      // triangles of (p, q) edges where `r`s are the intersection indices, update
      // the number of triangles of the pairs (p, r) and (q, r).
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(intersection_indices.size()),
        update_edges_p_r_q_r_num_triangles<vertex_t, edge_t, decltype(edge_first)>{
          edgelist_srcs.size(),
          0,
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(), intersection_indices.size()),
          raft::device_span<edge_t>(num_triangles.data(), num_triangles.size()),
          edge_first});

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(intersection_indices.size()),
        update_edges_p_r_q_r_num_triangles<vertex_t, edge_t, decltype(edge_first)>{
          edgelist_srcs.size(),
          1,
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(), intersection_indices.size()),
          raft::device_span<edge_t>(num_triangles.data(), num_triangles.size()),
          edge_first});
    }
    printf("\ndone with the iteration\n");
    printf("\nafter updating p, r and q, r edges\n");
      raft::print_device_vector("num_triangles", num_triangles.data(), num_triangles.size(), std::cout);
    prev_chunk_size += chunk_size;
  }

  /*
  printf("\nfrom edge triangle count and size = %d\n", num_triangles.size());
  raft::print_device_vector("edgelist_srcs", edgelist_srcs.data(), edgelist_srcs.size(), std::cout);
  raft::print_device_vector("edgelist_dsts", edgelist_dsts.data(), edgelist_dsts.size(), std::cout);
  raft::print_device_vector("triangle_count", num_triangles.data(), num_triangles.size(), std::cout);
  printf("\n");
  */
  /*
  std::vector<rmm::device_uvector<edge_t>> buffer{};
  buffer.push_back(std::move(num_triangles));
  auto buff_counts =
    edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t>(std::move(buffer));
  */

  //std::vector<rmm::device_uvector<edge_t>> buffer{};
  //buffer.push_back(std::move(num_triangles));
  //buffer.reserve(num_triangles.size());
  //buffer.push_back(std::move(num_triangles));
  //printf("\nother count\n");
  //raft::print_device_vector("triangle_count", buffer[0].data(), buffer[0].size(), std::cout);
  printf("\n");


  //auto buff_counts =
  //  edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t>(std::move(buffer));
  
  //#if 0
  cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t> counts(
    handle, graph_view);

  /*
  auto counts_ = detail::edge_partition_edge_property_device_view_t<edge_t, edge_t const*>(
                       (buff_counts.view()), 0);
  */

  //edge_t*
  //auto y = x.value_first();
  //raft::print_device_vector("prop_triangle_ct", counts_.value_first(), num_triangles.size(), std::cout);
  /*
  cugraph::transform_e(
    handle,
    graph_view,
    // buccket.
    cugraph::edge_src_dummy_property_t{}.view(),
    cugraph::edge_dst_dummy_property_t{}.view(),
    buff_counts.view(),
    [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto count) {
      //printf("\nedge %d, %d, count = %d\n", src, dst, count);
      return count;
    },
    counts.mutable_view(),
    false);
  */
  

  cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> valid_edges(handle);
      valid_edges.insert(edgelist_srcs.begin(),
                         edgelist_srcs.end(),
                         edgelist_dsts.begin());

  auto cur_graph_view = graph_view;
  /*
  auto unmasked_cur_graph_view = cur_graph_view;
  if (unmasked_cur_graph_view.has_edge_mask()) { unmasked_cur_graph_view.clear_edge_mask(); }
  */

  auto edge_last = edge_first + edgelist_srcs.size();
  printf("\nthe number of edges = %d\n", edgelist_srcs.size());
  cugraph::transform_e(
    handle,
    graph_view,
    valid_edges,
    cugraph::edge_src_dummy_property_t{}.view(),
    cugraph::edge_dst_dummy_property_t{}.view(),
    cugraph::edge_dummy_property_t{}.view(),
    [edge_first,
     edge_last,
     num_edges = edgelist_srcs.size(),
     num_triangles = num_triangles.data()] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {
      //printf("\nedge %d, %d\n", src, dst);
      auto pair = thrust::make_tuple(src, dst);
      // Find its position in 'edges'

      auto itr_pair =
        thrust::lower_bound(thrust::seq, edge_first, edge_last, pair);
      //auto itr_pair = thrust::lower_bound(thrust::seq, edge_first, edge_last, pair);
      //assert(*itr_p_r_q_r == p_r_q_r_pair);
      //if (itr_pair != edge_last && *itr_pair == pair)  {
      auto idx_pair = thrust::distance(edge_first, itr_pair);
      printf("\nin - edge %d, %d, count = %d\n", src, dst, num_triangles[idx_pair]);
      return num_triangles[idx_pair];
      //}
    },
    counts.mutable_view(),
    false);
  

  //#endif

  return counts;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t> edge_triangle_count(
  raft::handle_t const& handle, graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view)
{
  return detail::edge_triangle_count_impl(handle, graph_view);
}

}  // namespace cugraph
