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

template <typename vertex_t, typename edge_t, typename EdgeIterator, bool is_q_r_edge, bool multi_gpu>
// difference something. 
edge_t remove_overcompensating_edges(raft::handle_t const& handle,
                                     size_t buffer_size,
                                     EdgeIterator set_a_query_edges, // (p, q) edges
                                     EdgeIterator set_b_query_edges,
                                     // rename querry_edge_first
                                     // rename querry_edge_last
                                     raft::device_span<vertex_t const> set_c_weak_edges_srcs, // FIXME: rename this, no need for first
                                     raft::device_span<vertex_t const> set_c_weak_edges_dsts,
                                     std::vector<vertex_t> vertex_partition_range_lasts) // FIXME: rename this
{

  // To avoid over-compensating, check whether the 'potential_closing_edges'
  // are within the weak edges. If yes, the was already unrolled

  // FIXME: thrust::set_difference for SG
  // set_difference once for major or minor comm
  // rename set_A_last and set B. finding the difference
  // Make it more general, not k-truss oriented

  rmm::device_uvector<vertex_t> set_a_query_edges_srcs(buffer_size, handle.get_stream());
  rmm::device_uvector<vertex_t> set_a_query_edges_dsts(buffer_size, handle.get_stream());
  std::vector<size_t> rx_count{};


  if constexpr (multi_gpu) {

    
    // FIXME: Just zip src and dst to copy at once for the edges
    thrust::copy(handle.get_thrust_policy(),
                set_a_query_edges,
                set_a_query_edges + buffer_size,
                thrust::make_zip_iterator(set_a_query_edges_srcs.begin(), set_a_query_edges_dsts.begin()));

    auto& comm           = handle.get_comms();
    auto const comm_rank = comm.get_rank();



    // group_by_count to get the destination of each edges
    std::tie(set_a_query_edges_srcs, set_a_query_edges_dsts, std::ignore, std::ignore, std::ignore, rx_count) =
        detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       float,
                                                                                       int32_t>(
          handle, std::move(set_a_query_edges_srcs), std::move(set_a_query_edges_dsts), std::nullopt, std::nullopt, std::nullopt, vertex_partition_range_lasts);


    rmm::device_uvector<vertex_t> has_edge(set_a_query_edges_srcs.size(), handle.get_stream()); // type should be size_t

    auto set_c_weak_edges_first = thrust::make_zip_iterator(set_c_weak_edges_srcs.begin(), set_c_weak_edges_dsts.begin());  // setBedges
    auto set_c_weak_edges_last  = thrust::make_zip_iterator(set_c_weak_edges_srcs.end(), set_c_weak_edges_dsts.end());
    auto set_a_query_edges_first = thrust::make_zip_iterator(set_a_query_edges_srcs.begin(), set_a_query_edges_dsts.begin());

    /*
    thrust::transform(
      handle.get_thrust_policy()
      ...
    )
    */

    thrust::tabulate(
            handle.get_thrust_policy(),
            has_edge.begin(), // FIXME: Properly reconstruct (p, r) even when there is no overcompensation ************************************
            has_edge.end(),
            [
              set_c_weak_edges_first,
              set_c_weak_edges_last,
              set_a_query_edges_first
            ] __device__(auto i) {
              return thrust::binary_search(
                thrust::seq, set_c_weak_edges_first, set_c_weak_edges_last, set_a_query_edges_first[i]);
              });
    

    if (comm_rank == 1) {
      raft::print_device_vector("has_edge_b_s_v", has_edge.data(), has_edge.size(), std::cout);
    }

    std::tie(has_edge, std::ignore) =
      shuffle_values(handle.get_comms(), has_edge.begin(), rx_count, handle.get_stream());
    

    //if (comm_rank == 0) {
      raft::print_device_vector("has_edge_a_s_v", has_edge.data(), has_edge.size(), std::cout);
    //}



    if (comm_rank == 1) {
      
      raft::print_device_vector("set_c_weak_edges_srcs", set_c_weak_edges_srcs.data(), set_c_weak_edges_srcs.size(), std::cout);
      raft::print_device_vector("set_c_weak_edges_dsts", set_c_weak_edges_dsts.data(), set_c_weak_edges_dsts.size(), std::cout);

      raft::print_device_vector("set_a_query_edges_srcs", set_a_query_edges_srcs.data(), set_a_query_edges_srcs.size(), std::cout);
      raft::print_device_vector("set_a_query_edges_dsts", set_a_query_edges_dsts.data(), set_a_query_edges_dsts.size(), std::cout);
    }

    // FIXME: thrust::remove_if (resize). No need for sort_by_key and upper_bound

    thrust::sort_by_key(handle.get_thrust_policy(),
            has_edge.begin(),
            has_edge.end(),
            thrust::make_zip_iterator(set_a_query_edges,
                                      set_b_query_edges)
            );
    
    //raft::print_device_vector("sorted_has_edge_a_s_v", has_edge.data(), has_edge.size(), std::cout);
    
    // thrust upper_bound 0 
    auto itr = thrust::upper_bound(
          handle.get_thrust_policy(), has_edge.begin(), has_edge.end(), vertex_t{0});
    
    // FIXME: No need to reconstruct the third array because we can zip all 3 edges of the triangle

    //printf("\nnumber of potential weak edges = %d\n", has_edge.size());
    auto dist = thrust::distance(has_edge.begin(), itr); // FIXME: Check whether -1 is necessary
  
    printf("\ndistance = %d\n", dist);

    return dist;
    //return 0;
    
  



  } else {
    auto edges_not_overcomp = thrust::remove_if(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(set_a_query_edges,
                                set_b_query_edges),
      thrust::make_zip_iterator(set_a_query_edges + buffer_size,
                                set_b_query_edges + buffer_size),
      [set_c_weak_edges_first =
        thrust::make_zip_iterator(set_c_weak_edges_srcs.begin(), set_c_weak_edges_dsts.begin()),
      set_c_weak_edges_last = thrust::make_zip_iterator(set_c_weak_edges_srcs.end(),
                                                set_c_weak_edges_dsts.end())] __device__(auto e) {
        auto potential_edge = thrust::get<0>(e);
        auto potential_or_incoming_edge = thrust::make_tuple(thrust::get<0>(potential_edge), thrust::get<1>(potential_edge));
        if constexpr (is_q_r_edge) {
          potential_or_incoming_edge = thrust::make_tuple(thrust::get<1>(potential_edge), thrust::get<0>(potential_edge));
        };

        /*
        auto itr = thrust::lower_bound(
          thrust::seq, set_c_weak_edges_first, set_c_weak_edges_last, potential_or_incoming_edge);
        */
        return thrust::binary_search(
          thrust::seq, set_c_weak_edges_first, set_c_weak_edges_last, potential_or_incoming_edge);
      });

    auto dist = thrust::distance(thrust::make_zip_iterator(set_a_query_edges,
                                                          set_b_query_edges),
                                edges_not_overcomp);

    printf("\nlegacy - distance = %d\n", dist);
    return dist;
  }
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
struct extract_edges_and_triangle_counts {
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

    //printf("\nsrc = %d, dst = %d, idx = %d\n", thrust::get<0>(tagged_src), dst, thrust::get<1>(tagged_src));
    /*
    if (thrust::get<0>(tagged_src) == 3) {
      printf("\nsrc = 3, dst = %d, idx = %d\n", dst, thrust::get<1>(tagged_src));
    }
    */
    return thrust::make_optional(thrust::make_tuple(dst, thrust::get<1>(tagged_src)));
  }
};

template <typename vertex_t, typename edge_t, typename EdgeIterator, bool multi_gpu>
struct extract_q_idx_closing {
  using return_type = thrust::optional<thrust::tuple<vertex_t, vertex_t, vertex_t, edge_t>>;
  EdgeIterator major_weak_edgelist_dsts_tag_first{};
  EdgeIterator major_weak_edgelist_dsts_tag_last{};
  raft::device_span<vertex_t const> major_weak_edgelist_srcs{};
  raft::device_span<vertex_t const> weak_edgelist_dsts{};
  raft::device_span<edge_t const> weak_edgelist_tags{}; // FIXME: keep this when performing chunking

  return_type __device__ operator()(thrust::tuple<vertex_t, edge_t> tagged_src,
                                    vertex_t dst,
                                    thrust::nullopt_t,
                                    thrust::nullopt_t,
                                    thrust::nullopt_t) const
  {

    auto itr = thrust::lower_bound(
        thrust::seq,
        major_weak_edgelist_dsts_tag_first,
        major_weak_edgelist_dsts_tag_last,
        thrust::make_tuple(dst, thrust::get<1>(tagged_src)));
      
    auto idx = thrust::distance(major_weak_edgelist_dsts_tag_first, itr);
    
    return (itr != major_weak_edgelist_dsts_tag_last && *itr == thrust::make_tuple(dst, thrust::get<1>(tagged_src)))
            ? thrust::make_optional(thrust::make_tuple(thrust::get<0>(tagged_src), dst, major_weak_edgelist_srcs[idx], thrust::get<1>(tagged_src)))
            : thrust::nullopt;


    /*
    edge_t idx = thrust::get<1>(tagged_src);
    return dst == weak_edgelist_dsts[idx]
             ? thrust::make_optional(thrust::make_tuple(thrust::get<0>(tagged_src), idx))
             : thrust::nullopt;
    */
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
  EdgeIterator weak_edge_first{};
  EdgeIterator weak_edge_dst_tag_first{};
  EdgeIterator weak_edge_dst_tag_last{};
  EdgeIterator closing_r_tag{};

  raft::device_span<edge_t const> weak_edge_idx{};
  raft::device_span<edge_t const>  chunk_global_weak_edgelist_tags{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    // FIXME: When performing chunking, run binary search on the idx of the weak
    // edges for multi-GPU. similar to the example in 'extract_q_idx_closing'.
    // 1) Have the pair(weak_edges, tag) sorted by (tag)
    // 2) Perform a binary search on the tag to find if the weak edge exist in the selected
    //    chunk
    // 3) if it exists, return a pair, otherwise, return a thrust::nullopt
    
    /*
    auto itr = thrust::lower_bound(
      thrust::seq, chunk_global_weak_edgelist_tags.begin(), chunk_global_weak_edgelist_tags.end(), weak_edge_idx[i]);
    */

    auto itr = thrust::lower_bound(
      thrust::seq, weak_edge_dst_tag_first, weak_edge_dst_tag_last, closing_r_tag[i]);

    
    auto idx = thrust::distance(weak_edge_dst_tag_first, itr);
    
    return *(weak_edge_first + idx);
    
    
    
    
    //return *(weak_edge + weak_edge_idx[i]);
    
  }
};

template <typename vertex_t, typename edge_t, typename EdgeIterator, bool generate_p_q>
struct generate_p_q_q_r {
  EdgeIterator weak_edge{};
  raft::device_span<vertex_t const> q_closing{};
  raft::device_span<edge_t const> weak_edge_idx{};
  raft::device_span<edge_t const>  chunk_global_weak_edgelist_tags{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    // FIXME: When performing chunking, run binary search on the idx of the weak
    // edges for multi-GPU. similar to the example in 'extract_q_idx_closing'.
    // 1) Have the pair(weak_edges, tag) sorted by (tag)
    // 2) Perform a binary search on the tag to find if the weak edge exist in the selected
    //    chunk
    // 3) if it exists, return a pair, otherwise, return a thrust::nullopt

    /*
    auto itr = thrust::lower_bound(
        thrust::seq, chunk_global_weak_edgelist_tags.begin(), chunk_global_weak_edgelist_tags.end(), weak_edge_idx[i]);
      
    auto idx = thrust::distance(chunk_global_weak_edgelist_tags.begin(), itr);
    
    if constexpr (generate_p_q) {
      return thrust::make_tuple(thrust::get<0>(*(weak_edge + idx)), q_closing[i]);
      // FIXME: If single GPU, return this. Add template for multi_gpu
    } else {
      return thrust::make_tuple(q_closing[i], thrust::get<1>(*(weak_edge + idx)));
    }
    */
    

    
    if constexpr (generate_p_q) {
      return thrust::make_tuple(thrust::get<0>(*(weak_edge + weak_edge_idx[i])), q_closing[i]);
    } else {
      return thrust::make_tuple(q_closing[i], thrust::get<1>(*(weak_edge + weak_edge_idx[i])));
    }
    
    
  }
};

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void decrease_triangle_count(raft::handle_t const& handle,
                          graph_view_t<vertex_t, edge_t, false, multi_gpu> & cur_graph_view,
                          edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t> & edge_triangle_counts,
                          raft::device_span<vertex_t> edge_srcs,
                          raft::device_span<vertex_t> edge_dsts
                          ) {
  
  // Before updating the count, we need to clear the mask
  // cur_graph_view.clear_edge_mask();
  auto vertex_pair_buffer_begin = thrust::make_zip_iterator(edge_srcs.begin(), edge_dsts.begin());
  
  thrust::sort(handle.get_thrust_policy(),
               vertex_pair_buffer_begin,
               vertex_pair_buffer_begin + edge_srcs.size());
  
  auto unique_pair_count = thrust::unique_count(handle.get_thrust_policy(),
                                                vertex_pair_buffer_begin,
                                                vertex_pair_buffer_begin + edge_srcs.size());
  
  rmm::device_uvector<edge_t> decrease_count(unique_pair_count, handle.get_stream());

  /*
  rmm::device_uvector<edge_t> decrease_count_tmp(edge_srcs.size(),
                                                 handle.get_stream());
  
  thrust::fill(handle.get_thrust_policy(),
               decrease_count_tmp.begin(),
               decrease_count_tmp.end(),
               size_t{1});
  */
  
  auto vertex_pair_buffer_unique = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        unique_pair_count, handle.get_stream());
  
  thrust::reduce_by_key(handle.get_thrust_policy(),
                        vertex_pair_buffer_begin,
                        vertex_pair_buffer_begin + edge_srcs.size(),
                        thrust::make_constant_iterator(size_t{1}),
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
    edge_triangle_counts.view(),
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
    edge_triangle_counts.mutable_view(),
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

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  //printf("\nin 'accumulate_triangles_p_q_or_q_r' and size = %d\n", weak_edgelist_srcs.size());

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

  //std::cout<< "The intersection size for (p, q) or (q, r) edges = " << intersection_indices.size() << std::endl;
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
  std::optional<rmm::device_uvector<weight_t>> wgts{std::nullopt};

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
      std::tie(srcs, dsts, std::ignore, std::ignore, std::ignore, std::ignore) =
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

    edge_weight_view =
      edge_weight ? std::make_optional((*edge_weight).view())
                  : std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt};
    
    auto edge_triangle_counts = edge_triangle_count<vertex_t, edge_t, multi_gpu>(handle, cur_graph_view);

    cugraph::edge_property_t<decltype(cur_graph_view), bool> edge_mask(handle, cur_graph_view);
    cugraph::fill_edge_property(handle, cur_graph_view, true, edge_mask);

    while (true) {

      // extract the edges that have counts less than k - 2. Those edges will be unrolled
      auto [weak_edgelist_srcs, weak_edgelist_dsts] = extract_transform_e(handle,
                                                                          cur_graph_view,
                                                                          edge_src_dummy_property_t{}.view(),
                                                                          edge_dst_dummy_property_t{}.view(),
                                                                          edge_triangle_counts.view(),
                                                                          extract_weak_edges<vertex_t, edge_t>{k});

      
      auto num_weak_edges = weak_edgelist_srcs.size();
      if constexpr (multi_gpu) {
        num_weak_edges = host_scalar_allreduce(handle.get_comms(), num_weak_edges, raft::comms::op_t::SUM, handle.get_stream());
      }
      if (num_weak_edges == 0) { break; }
            auto weak_edgelist_first = thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin());
            thrust::sort(handle.get_thrust_policy(),
                                weak_edgelist_first,
                                weak_edgelist_first + weak_edgelist_srcs.size());
      
      // Find intersection edges
      size_t prev_chunk_size         = 0;
      size_t chunk_num_weak_edges = weak_edgelist_srcs.size();
      size_t edges_to_intersect_per_iteration =
            static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 17);
    

      auto num_chunks =
        raft::div_rounding_up_safe(weak_edgelist_srcs.size(), edges_to_intersect_per_iteration);

      if constexpr (multi_gpu) {
        num_chunks = host_scalar_allreduce(handle.get_comms(), num_chunks, raft::comms::op_t::SUM, handle.get_stream());
      }
      
      printf("\nnum_chunks = %d\n", num_chunks);
      
      // FIXME: In case some ranks have no weak edges to process
      // Or simply in the for loop set i <= 0 but need to make sure the chunking process is not broken
      /*
      if (num_chunks == 0) {
        num_chunks = 1;
      }
      */
      
      for (size_t i = 0; i < num_chunks; ++i) {
        printf("\n in for loop chunk, i = %d, num_chunks = %d\n", i, num_chunks);
        auto chunk_size = std::min(edges_to_intersect_per_iteration, chunk_num_weak_edges);
        //RAFT_CUDA_TRY(cudaDeviceSynchronize());
        //printf("\ntracking hang\n");

        //auto x = raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size());
        //auto x = weak_edgelist_srcs.size();
        //RAFT_CUDA_TRY(cudaDeviceSynchronize());
        //printf("\nafter tracking hang, size = %d\n", x);

        //#if 0
        auto [vertex_pair_buffer_p_q, vertex_pair_buffer_p_r_edge_p_q, vertex_pair_buffer_q_r_edge_p_q] = accumulate_triangles_p_q_or_q_r<vertex_t, edge_t, weight_t, decltype(allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(size_t{0}, handle.get_stream())), multi_gpu>(
                    handle,
                    cur_graph_view,
                    raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
                    raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size()),
                    prev_chunk_size,
                    chunk_size, 
                    do_expensive_check);
        
        //#if 0
        //raft::print_device_vector("vertex_pair_buffer_p_q_srcs", std::get<0>(vertex_pair_buffer_p_q).data(), std::get<0>(vertex_pair_buffer_p_q).size(), std::cout);
        //raft::print_device_vector("vertex_pair_buffer_p_q_dsts", std::get<1>(vertex_pair_buffer_p_q).data(), std::get<1>(vertex_pair_buffer_p_q).size(), std::cout);

        //raft::print_device_vector("vertex_pair_buffer_p_r_edge_p_q_srcs", std::get<0>(vertex_pair_buffer_p_r_edge_p_q).data(), std::get<0>(vertex_pair_buffer_p_r_edge_p_q).size(), std::cout);
        //raft::print_device_vector("vertex_pair_buffer_p_r_edge_p_q_dsts", std::get<1>(vertex_pair_buffer_p_r_edge_p_q).data(), std::get<1>(vertex_pair_buffer_p_r_edge_p_q).size(), std::cout);
    
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_r_edge_p_q_srcs(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_r_edge_p_q_dsts(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_q_r_edge_p_q_srcs(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_q_r_edge_p_q_dsts(0, handle.get_stream());
        // Shuffle edges
        if constexpr (multi_gpu) {
          // FIXME: Check whether we need to shuffle (p, q) edges
          std::tie(vertex_pair_buffer_p_r_edge_p_q_srcs, vertex_pair_buffer_p_r_edge_p_q_dsts, std::ignore, std::ignore, std::ignore, std::ignore) =
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

          std::tie(vertex_pair_buffer_q_r_edge_p_q_srcs, vertex_pair_buffer_q_r_edge_p_q_dsts, std::ignore, std::ignore, std::ignore, std::ignore) =
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
        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        /*
        std::cout << "Unrolling (p, q) edges" << std::endl;
      
        raft::print_device_vector("vertex_pair_buffer_p_q", std::get<0>(vertex_pair_buffer_p_q).data(), std::get<0>(vertex_pair_buffer_p_q).size(), std::cout);
        raft::print_device_vector("vertex_pair_buffer_p_q", std::get<1>(vertex_pair_buffer_p_q).data(), std::get<1>(vertex_pair_buffer_p_q).size(), std::cout);
        printf("\n");
        raft::print_device_vector("vertex_pair_buffer_p_r_edge_p_q_srcs", vertex_pair_buffer_p_r_edge_p_q_srcs.data(), vertex_pair_buffer_p_r_edge_p_q_srcs.size(), std::cout);
        raft::print_device_vector("vertex_pair_buffer_p_r_edge_p_q_dsts", vertex_pair_buffer_p_r_edge_p_q_dsts.data(), vertex_pair_buffer_p_r_edge_p_q_dsts.size(), std::cout);
        printf("\n");
        raft::print_device_vector("vertex_pair_buffer_q_r_edge_p_q_srcs", vertex_pair_buffer_q_r_edge_p_q_srcs.data(), vertex_pair_buffer_q_r_edge_p_q_srcs.size(), std::cout);
        raft::print_device_vector("vertex_pair_buffer_q_r_edge_p_q_dsts", vertex_pair_buffer_q_r_edge_p_q_dsts.data(), vertex_pair_buffer_q_r_edge_p_q_dsts.size(), std::cout);
        */

        decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
          handle,
          cur_graph_view,
          edge_triangle_counts,
          raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_q).data(), std::get<0>(vertex_pair_buffer_p_q).size()),
          raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_p_q).data(), std::get<1>(vertex_pair_buffer_p_q).size())
        );

        decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            edge_triangle_counts,
            multi_gpu ? raft::device_span<vertex_t>(vertex_pair_buffer_p_r_edge_p_q_srcs.data(), vertex_pair_buffer_p_r_edge_p_q_srcs.size()) : raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_r_edge_p_q).data(), std::get<0>(vertex_pair_buffer_p_r_edge_p_q).size()), // FIXME: Make sure multi_gpu is properly handles
            multi_gpu ? raft::device_span<vertex_t>(vertex_pair_buffer_p_r_edge_p_q_dsts.data(), vertex_pair_buffer_p_r_edge_p_q_dsts.size()) : raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_r_edge_p_q).data(), std::get<0>(vertex_pair_buffer_p_r_edge_p_q).size()) // FIXME: Make sure multi_gpu is properly handles
          );  

        decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            edge_triangle_counts,
            multi_gpu ? raft::device_span<vertex_t>(vertex_pair_buffer_q_r_edge_p_q_srcs.data(), vertex_pair_buffer_q_r_edge_p_q_srcs.size()) : raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_q_r_edge_p_q).data(), std::get<0>(vertex_pair_buffer_q_r_edge_p_q).size()),
            multi_gpu ? raft::device_span<vertex_t>(vertex_pair_buffer_q_r_edge_p_q_dsts.data(), vertex_pair_buffer_q_r_edge_p_q_dsts.size()) : raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_q_r_edge_p_q).data(), std::get<0>(vertex_pair_buffer_q_r_edge_p_q).size())
          );
        
        prev_chunk_size += chunk_size;
        chunk_num_weak_edges -= chunk_size;

        //#endif
      }
      
      //#if 0
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      //printf("\nafter unrolling (p, q) edges\n");
      auto [srcs_0, dsts_0, count_0] = extract_transform_e(handle,
                                                    cur_graph_view,
                                                    cugraph::edge_src_dummy_property_t{}.view(),
                                                    cugraph::edge_dst_dummy_property_t{}.view(),
                                                    //view_concat(edge_triangle_counts.view(), modified_triangle_count.view()),
                                                    edge_triangle_counts.view(),
                                                    extract_edges_and_triangle_counts<vertex_t, edge_t>{});
      /*
      raft::print_device_vector("unrolled_srcs", srcs_0.data(), srcs_0.size(), std::cout);
      raft::print_device_vector("unrolled_dsts", dsts_0.data(), dsts_0.size(), std::cout);
      raft::print_device_vector("unrolled_n_tr", count_0.data(), count_0.size(), std::cout);
      */
      


      //#if 0
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
            
      thrust::sort(handle.get_thrust_policy(), unique_weak_edgelist_srcs.begin(), unique_weak_edgelist_srcs.end()); // No need to sort the 'dst' since they are already sorted
      
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

      auto vertex_q_r_end = thrust::set_union(handle.get_thrust_policy(),
                                        unique_weak_edgelist_srcs.begin(),
                                        unique_weak_edgelist_srcs.end(),
                                        unique_weak_edgelist_dsts.begin(),
                                        unique_weak_edgelist_dsts.end(),
                                        vertex_q_r.begin());
      
      vertex_q_r.resize(thrust::distance(vertex_q_r.begin(), vertex_q_r_end), handle.get_stream());

      thrust::sort(handle.get_thrust_policy(), vertex_q_r.begin(), vertex_q_r.end());

      auto weak_unique_v_end =  thrust::unique(
                                      handle.get_thrust_policy(),
                                      vertex_q_r.begin(),
                                      vertex_q_r.end());
      
      vertex_q_r.resize(thrust::distance(vertex_q_r.begin(), weak_unique_v_end), handle.get_stream());

      // FIXME: perform all to all 'vertex_q_r'. ********************************
      // FIXME: Might not be able to perform this in chunk for MG
      // e.g: giving 4 weak edges where 2 belongs to the same triangle. If we were to process
      // each of these edges in different batches, we might not be able to find a triangle
      // Need a view of the whole graph to find triangles. But why was batching working in SG with 
      // nbr_intersection in case 1 ?(because I already have a view of the whole graph). For
      // case 2 we can still process the edges in batches as long as we already created the full csc graph
      // Isn't better to just create the csc graph with all edges at this point. 
      // Cannot create partial CSR, need the full one. Can we create ehe CSC in chunks by adding set of
      // edges at a time?

      if constexpr (multi_gpu) {
        auto& comm           = handle.get_comms();
        auto const comm_rank = comm.get_rank();  // FIXME: for debugging
        // Get global weak_edgelist
        // FIXME: Perform all-to-all in chunks
        auto global_vertex_q_r = cugraph::detail::device_allgatherv(
          handle, comm, raft::device_span<vertex_t const>(vertex_q_r.data(), vertex_q_r.size()));
        
        thrust::sort(handle.get_thrust_policy(), global_vertex_q_r.begin(), global_vertex_q_r.end());

        weak_unique_v_end = thrust::unique(
                                  handle.get_thrust_policy(),
                                  global_vertex_q_r.begin(),
                                  global_vertex_q_r.end());
      
        global_vertex_q_r.resize(thrust::distance(global_vertex_q_r.begin(), weak_unique_v_end), handle.get_stream());

        //raft::print_device_vector("1_global_vertex_q_r", global_vertex_q_r.data(), global_vertex_q_r.size(), std::cout);

        // FIXME: Can be very expensive and increase peak memory
        vertex_q_r.resize(global_vertex_q_r.size(), handle.get_stream());
    
        thrust::copy(
          handle.get_thrust_policy(),
          global_vertex_q_r.begin(),
          global_vertex_q_r.end(),
          vertex_q_r.begin());
      }

      //raft::print_device_vector("2_vertex_q_r", vertex_q_r.data(), vertex_q_r.size(), std::cout);

      weak_edgelist_first = thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin()); // FIXME: is this necessary ?
      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      //std::cout << "before extracting edges" << std::endl;
      auto [srcs_to_q_r, dsts_to_q_r] = extract_transform_e(handle,
                                                cur_graph_view,
                                                cugraph::edge_src_dummy_property_t{}.view(),
                                                cugraph::edge_dst_dummy_property_t{}.view(),
                                                cugraph::edge_dummy_property_t{}.view(),
                                                extract_edges_to_q_r<vertex_t>{raft::device_span<vertex_t const>(vertex_q_r.data(), vertex_q_r.size())});
      

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      /*
      std::cout << "arrays to be shuffled" << std::endl;
      raft::print_device_vector("vertex_q_r", vertex_q_r.data(), vertex_q_r.size(), std::cout);
      //raft::print_device_vector("vertex_q_r", vertex_q_r.data(), vertex_q_r.size(), std::cout);
      raft::print_device_vector("srcs_to_q_r", srcs_to_q_r.data(), srcs_to_q_r.size(), std::cout);
      raft::print_device_vector("dsts_to_q_r", dsts_to_q_r.data(), dsts_to_q_r.size(), std::cout);
      */
      if constexpr (multi_gpu) {
        std::tie(dsts_to_q_r, srcs_to_q_r, std::ignore, std::ignore, std::ignore, std::ignore) =
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

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      //printf("\ndone shuffling\n");
      
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

      rmm::device_uvector<vertex_t> renumbered_weak_edgelist_srcs(0, handle.get_stream());
      rmm::device_uvector<vertex_t> renumbered_weak_edgelist_dsts(0, handle.get_stream());
      size_t weak_edgelist_size = weak_edgelist_srcs.size();

      if constexpr (multi_gpu) {
        renumbered_weak_edgelist_srcs.resize(weak_edgelist_srcs.size(), handle.get_stream());
        renumbered_weak_edgelist_dsts.resize(weak_edgelist_dsts.size(), handle.get_stream());

        thrust::copy(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin()),
          thrust::make_zip_iterator(weak_edgelist_srcs.end(), weak_edgelist_dsts.end()),
          thrust::make_zip_iterator(renumbered_weak_edgelist_srcs.begin(), renumbered_weak_edgelist_dsts.begin()));

        rmm::device_uvector<vertex_t> shuffled_weak_edgelist_srcs{0, handle.get_stream()};
        rmm::device_uvector<vertex_t> shuffled_weak_edgelist_dsts{0, handle.get_stream()};

        std::tie(
          renumbered_weak_edgelist_srcs, renumbered_weak_edgelist_dsts, std::ignore, std::ignore, std::ignore, std::ignore) =
          detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                         edge_t,
                                                                                         weight_t,
                                                                                         int32_t>(
            handle,
            std::move(renumbered_weak_edgelist_srcs),
            std::move(renumbered_weak_edgelist_dsts),
            std::nullopt,
            std::nullopt,
            std::nullopt);

        renumber_ext_vertices<vertex_t, multi_gpu>(
          handle,
          renumbered_weak_edgelist_srcs.data(),
          renumbered_weak_edgelist_srcs.size(),
          (*renumber_map_q_r).data(),
          csc_q_r_graph_view.local_vertex_partition_range_first(),
          csc_q_r_graph_view.local_vertex_partition_range_last(),
          true);

        renumber_ext_vertices<vertex_t, multi_gpu>(
          handle,
          renumbered_weak_edgelist_dsts.data(),
          renumbered_weak_edgelist_dsts.size(),
          (*renumber_map_q_r).data(),
          csc_q_r_graph_view.local_vertex_partition_range_first(),
          csc_q_r_graph_view.local_vertex_partition_range_last(),
          true);
        
        weak_edgelist_size = renumbered_weak_edgelist_srcs.size();
        weak_edgelist_first =
        thrust::make_zip_iterator(renumbered_weak_edgelist_srcs.begin(), renumbered_weak_edgelist_dsts.begin());
        thrust::sort(handle.get_thrust_policy(),
                    weak_edgelist_first,
                    weak_edgelist_first + renumbered_weak_edgelist_srcs.size());
      }
      prev_chunk_size         = 0;
      chunk_num_weak_edges = weak_edgelist_size;

      // FIXME: No need to recompute this. It's the same value as above when unrolling (p, q) edges
      num_chunks =
        raft::div_rounding_up_safe(weak_edgelist_size, edges_to_intersect_per_iteration);
      
      if (num_chunks == 0) {
        num_chunks = 1;
      }
      
      //auto sorted_weak_edgelist_srcs = thrust::get<0>(weak_edgelist_first.get_iterator_tuple()); // FIXME: Remove this
      //auto sorted_weak_edgelist_dsts = thrust::get<1>(weak_edgelist_first.get_iterator_tuple());


      for (size_t i = 0; i < num_chunks; ++i) {
        auto chunk_size = std::min(edges_to_intersect_per_iteration, chunk_num_weak_edges);
  
        // Find intersection of weak edges
        auto [vertex_pair_buffer_q_r, vertex_pair_buffer_p_q_edge_q_r, vertex_pair_buffer_p_r_edge_q_r] = accumulate_triangles_p_q_or_q_r<vertex_t, edge_t, weight_t, decltype(allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(size_t{0}, handle.get_stream())), multi_gpu>(
                    handle,
                    csc_q_r_graph_view,
                    //raft::device_span<vertex_t const>(sorted_weak_edgelist_srcs, weak_edgelist_size),
                    //raft::device_span<vertex_t const>(sorted_weak_edgelist_dsts, weak_edgelist_size),
                    raft::device_span<vertex_t const>(renumbered_weak_edgelist_srcs.data(), renumbered_weak_edgelist_srcs.size()),
                    raft::device_span<vertex_t const>(renumbered_weak_edgelist_dsts.data(), renumbered_weak_edgelist_dsts.size()),
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
          
          //printf("\ndebugging (q, r) edges unrolling\n");

          //raft::print_device_vector("vertex_pair_buffer_q_r_srcs_b_u", std::get<0>(vertex_pair_buffer_q_r).data(), std::get<0>(vertex_pair_buffer_q_r).size(), std::cout);
          //raft::print_device_vector("vertex_pair_buffer_q_r_dsts_b_u", std::get<1>(vertex_pair_buffer_q_r).data(), std::get<1>(vertex_pair_buffer_q_r).size(), std::cout);
          //printf("(q, r) edge size = %d\n", std::get<0>(vertex_pair_buffer_q_r).size());

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
        }


        //printf("\ndebugging (q, r) edges unrolling\n");
        //raft::print_device_vector("vertex_pair_buffer_q_r_srcs_", std::get<0>(vertex_pair_buffer_q_r).data(), std::get<0>(vertex_pair_buffer_q_r).size(), std::cout);
        //raft::print_device_vector("vertex_pair_buffer_q_r_dsts_", std::get<1>(vertex_pair_buffer_q_r).data(), std::get<1>(vertex_pair_buffer_q_r).size(), std::cout);

        edge_t num_edges_not_overcomp = 0;

        if constexpr (multi_gpu) {

          // Get global weak edges
          // FIXME: Retrieve onlu a fraction of the weak edges.
        
          auto& comm           = handle.get_comms();
          auto const comm_rank = comm.get_rank();  // FIXME: for debugging
          // Get global weak_edgelist
          // FIXME: Perform all-to-all in chunks
          auto chunk_global_weak_edgelist_srcs = cugraph::detail::device_allgatherv(
            handle, comm, raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()));
          // FIXME: Perform all-to-all in chunks
          auto chunk_global_weak_edgelist_dsts = cugraph::detail::device_allgatherv(
            handle, comm, raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size()));

          
          //raft::print_device_vector("chunk_global_weak_edgelist_srcs", chunk_global_weak_edgelist_srcs.data(), chunk_global_weak_edgelist_srcs.size(), std::cout);
          //raft::print_device_vector("chunk_global_weak_edgelist_dsts", chunk_global_weak_edgelist_dsts.data(), chunk_global_weak_edgelist_dsts.size(), std::cout);
          
          
          // Sort the weak edges if they are not already
          auto chunk_global_weak_edgelist_first =
            thrust::make_zip_iterator(chunk_global_weak_edgelist_srcs.begin(), chunk_global_weak_edgelist_dsts.begin());
          thrust::sort(handle.get_thrust_policy(),
                       chunk_global_weak_edgelist_first,
                       chunk_global_weak_edgelist_first + chunk_global_weak_edgelist_srcs.size());
        

          num_edges_not_overcomp =
              remove_overcompensating_edges<vertex_t,
                                            edge_t,
                                            decltype(get_dataframe_buffer_begin(vertex_pair_buffer_q_r)),
                                            true,
                                            false // FIXME: Set it to False for now
                                            >(
                handle,
                size_dataframe_buffer(vertex_pair_buffer_p_q_edge_q_r),
                get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_q_r), // FIXME: cannot be a copy, needs to be the original one so overcompensatiing edges can be removed
                get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_q_r), // FIXME: cannot be a copy, needs to be the original one so overcompensatiing edges can be removed
                raft::device_span<vertex_t const>(chunk_global_weak_edgelist_srcs.data(), chunk_global_weak_edgelist_srcs.size()),
                raft::device_span<vertex_t const>(chunk_global_weak_edgelist_dsts.data(), chunk_global_weak_edgelist_dsts.size()),
                cur_graph_view.vertex_partition_range_lasts()
             );
          
          //std::cout << "num (q, r) edges after removing = " << num_edges_not_overcomp << std::endl;
          //printf("\n\n");
          resize_dataframe_buffer(vertex_pair_buffer_p_q_edge_q_r, num_edges_not_overcomp, handle.get_stream());
          resize_dataframe_buffer(vertex_pair_buffer_p_r_edge_q_r, num_edges_not_overcomp, handle.get_stream());

          // resize initial (q, r) edges
          // Note: Once chunking is implemented, reconstruct the (q, r) edges only outside
          // FIXME: No need to reconstruct the third array because we can zip all 3 edges of the triangle
          // of the chunk's 'for loop'
          resize_dataframe_buffer(vertex_pair_buffer_q_r, num_edges_not_overcomp, handle.get_stream());

          // FIXME: No need to reconstruct the third array because we can zip all 3 edges of the triangle
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

          

        } else {

          num_edges_not_overcomp =
            remove_overcompensating_edges<vertex_t,
                                          edge_t,
                                          decltype(get_dataframe_buffer_begin(vertex_pair_buffer_q_r)),
                                          true,
                                          false // FIXME: Set it to False for now
                                          >(
              handle,
              size_dataframe_buffer(vertex_pair_buffer_p_q_edge_q_r),
              get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_q_r), // FIXME: cannot be a copy, needs to be the original one so overcompensatiing edges can be removed
              get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_q_r), // FIXME: cannot be a copy, needs to be the original one so overcompensatiing edges can be removed
              raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
              raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size()),
              cur_graph_view.vertex_partition_range_lasts()
            );
          
          //std::cout << "num (q, r) edges after removing = " << num_edges_not_overcomp << std::endl;
          
          resize_dataframe_buffer(vertex_pair_buffer_p_q_edge_q_r, num_edges_not_overcomp, handle.get_stream());
          resize_dataframe_buffer(vertex_pair_buffer_p_r_edge_q_r, num_edges_not_overcomp, handle.get_stream());

          // resize initial (q, r) edges
          resize_dataframe_buffer(vertex_pair_buffer_q_r, num_edges_not_overcomp, handle.get_stream());
          
          // Reconstruct (q, r) edges that didn't already have their count updated
          // FIXME: No need to reconstruct the third array because we can zip all 3 edges of the triangle
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
        }

        if constexpr (multi_gpu) {        
          // Shuffle before updating count
          rmm::device_uvector<vertex_t> vertex_pair_buffer_q_r_srcs(0, handle.get_stream());
          rmm::device_uvector<vertex_t> vertex_pair_buffer_q_r_dsts(0, handle.get_stream());
        
          std::tie(vertex_pair_buffer_q_r_srcs, vertex_pair_buffer_q_r_dsts, std::ignore, std::ignore, std::ignore, std::ignore) =
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
      
          decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            edge_triangle_counts,
            raft::device_span<vertex_t>(vertex_pair_buffer_q_r_srcs.data(), vertex_pair_buffer_q_r_srcs.size()),
            raft::device_span<vertex_t>(vertex_pair_buffer_q_r_dsts.data(), vertex_pair_buffer_q_r_dsts.size())
          );

          // Shuffle before updating count
          rmm::device_uvector<vertex_t> vertex_pair_buffer_p_q_edge_q_r_srcs(0, handle.get_stream());
          rmm::device_uvector<vertex_t> vertex_pair_buffer_p_q_edge_q_r_dsts(0, handle.get_stream());
          
          std::tie(vertex_pair_buffer_p_q_edge_q_r_dsts, vertex_pair_buffer_p_q_edge_q_r_srcs, std::ignore, std::ignore, std::ignore, std::ignore) =
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

          decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            edge_triangle_counts,
            raft::device_span<vertex_t>(vertex_pair_buffer_p_q_edge_q_r_dsts.data(), vertex_pair_buffer_p_q_edge_q_r_dsts.size()),
            raft::device_span<vertex_t>(vertex_pair_buffer_p_q_edge_q_r_srcs.data(), vertex_pair_buffer_p_q_edge_q_r_srcs.size())
          );

          // Shuffle before updating count
          rmm::device_uvector<vertex_t> vertex_pair_buffer_p_r_edge_q_r_srcs(0, handle.get_stream());
          rmm::device_uvector<vertex_t> vertex_pair_buffer_p_r_edge_q_r_dsts(0, handle.get_stream());
          
          std::tie(vertex_pair_buffer_p_r_edge_q_r_dsts, vertex_pair_buffer_p_r_edge_q_r_srcs, std::ignore, std::ignore, std::ignore, std::ignore) =
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
    
          decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            edge_triangle_counts,
            raft::device_span<vertex_t>(vertex_pair_buffer_p_r_edge_q_r_dsts.data(), vertex_pair_buffer_p_r_edge_q_r_dsts.size()),
            raft::device_span<vertex_t>(vertex_pair_buffer_p_r_edge_q_r_srcs.data(), vertex_pair_buffer_p_r_edge_q_r_srcs.size())
          );
        
        } else {
          decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            edge_triangle_counts,
            raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_q_r).data(), std::get<0>(vertex_pair_buffer_q_r).size()),
            raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_q_r).data(), std::get<1>(vertex_pair_buffer_q_r).size())
          );
          decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            edge_triangle_counts,
            raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_p_q_edge_q_r).data(), std::get<0>(vertex_pair_buffer_p_q_edge_q_r).size()),
            raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_q_edge_q_r).data(), std::get<1>(vertex_pair_buffer_p_q_edge_q_r).size())
          );
          decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
            handle,
            cur_graph_view,
            edge_triangle_counts,
            raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_p_r_edge_q_r).data(), std::get<0>(vertex_pair_buffer_p_r_edge_q_r).size()),
            raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_r_edge_q_r).data(), std::get<1>(vertex_pair_buffer_p_r_edge_q_r).size())
          );
        }
        
        prev_chunk_size += chunk_size;
        chunk_num_weak_edges -= chunk_size;

      
      }



      //printf("\nafter unrolling (q, r) edges\n");
      auto [srcs_1, dsts_1, count_1] = extract_transform_e(handle,
                                                    cur_graph_view,
                                                    cugraph::edge_src_dummy_property_t{}.view(),
                                                    cugraph::edge_dst_dummy_property_t{}.view(),
                                                    //view_concat(edge_triangle_counts.view(), modified_triangle_count.view()),
                                                    edge_triangle_counts.view(),
                                                    extract_edges_and_triangle_counts<vertex_t, edge_t>{});
      
      /*
      raft::print_device_vector("unrolled_srcs", srcs_1.data(), srcs_1.size(), std::cout);
      raft::print_device_vector("unrolled_dsts", dsts_1.data(), dsts_1.size(), std::cout);
      raft::print_device_vector("unrolled_n_tr", count_1.data(), count_1.size(), std::cout);
      printf("\n");
      */

      /*
      std::cout<< "before zipping edgelist" << std::endl;
      raft::print_device_vector("sorted_weak_edgelist_srcs", sorted_weak_edgelist_srcs, weak_edgelist_size, std::cout);
      raft::print_device_vector("sorted_weak_edgelist_dsts", sorted_weak_edgelist_dsts, weak_edgelist_size, std::cout);
      */
      weak_edgelist_first =
            thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin());
      /*
      std::cout<< "after zipping edgelist" << std::endl;
      raft::print_device_vector("sorted_weak_edgelist_srcs", sorted_weak_edgelist_srcs, weak_edgelist_size, std::cout);
      raft::print_device_vector("sorted_weak_edgelist_dsts", sorted_weak_edgelist_dsts, weak_edgelist_size, std::cout);
      */
      //#if 0
      // Unrolling p, r edges
      // create pair weak_src, weak_edge_idx
      // create a dataframe buffer of size weak_edge_size
      // FIXME: No need to create a dataframe buffer. We can just zip weak_edgelist_srcs
      // with a vector counting from 0 .. 
    
      auto vertex_pair_buffer_p_tag =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, edge_t>>(weak_edgelist_srcs.size(),
                                                                    handle.get_stream());
      //#if 0  
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
          
          /*
          std::cout << "weak_edge_list size = " << weak_edgelist_srcs.size() << std::endl;
          raft::print_device_vector("sorted_weak_edgelist_srcs", sorted_weak_edgelist_srcs, weak_edgelist_size, std::cout);
          raft::print_device_vector("sorted_weak_edgelist_dsts", sorted_weak_edgelist_dsts, weak_edgelist_size, std::cout);
          raft::print_device_vector("weak_edgelist_srcs", weak_edgelist_srcs.data(), weak_edgelist_srcs.size(), std::cout);
          raft::print_device_vector("weak_edgelist_dsts", weak_edgelist_dsts.data(), weak_edgelist_dsts.size(), std::cout);
          */
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
      //#endif
      #if 0
      thrust::tabulate(
              handle.get_thrust_policy(),
              get_dataframe_buffer_begin(vertex_pair_buffer_p_tag),
              get_dataframe_buffer_end(vertex_pair_buffer_p_tag),
              [
                p = weak_edgelist_srcs.begin()
              ] __device__(auto idx) {
                return thrust::make_tuple(p[idx], idx);
                });
      #endif

      //raft::print_device_vector("edge_srcs", std::get<0>(vertex_pair_buffer_p_tag).data(), std::get<0>(vertex_pair_buffer_p_tag).size(), std::cout);
      //raft::print_device_vector("vertex_pair_buffer_tag", std::get<1>(vertex_pair_buffer_p_tag).data(), std::get<1>(vertex_pair_buffer_p_tag).size(), std::cout);
      
      vertex_frontier_t<vertex_t, edge_t, multi_gpu, false> vertex_frontier(handle, 1);
      rmm::device_uvector<edge_t> tag_cpy(std::get<1>(vertex_pair_buffer_p_tag).size(), handle.get_stream());
       thrust::copy(
          handle.get_thrust_policy(),
          std::get<1>(vertex_pair_buffer_p_tag).begin(),
          std::get<1>(vertex_pair_buffer_p_tag).end(),
          tag_cpy.begin());
      //std::cout << "emptying the vertex frontier" << std::endl;

      if constexpr (multi_gpu) {
        //printf("\nbefore shuffling\n");
        //raft::print_device_vector("b_vertex_pair_buffer_src", std::get<0>(vertex_pair_buffer_p_tag).data(), std::get<0>(vertex_pair_buffer_p_tag).size(), std::cout);
        //raft::print_device_vector("b_vertex_pair_buffer_tag", std::get<1>(vertex_pair_buffer_p_tag).data(), std::get<1>(vertex_pair_buffer_p_tag).size(), std::cout);

        // Shuffle vertices
        auto [p_vrtx, p_tag] =
              detail::shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
                handle,
                std::move(std::get<0>(vertex_pair_buffer_p_tag)),
                std::move(std::get<1>(vertex_pair_buffer_p_tag)),
                cur_graph_view.vertex_partition_range_lasts());

        //printf("\nafter shuffling\n");
        raft::print_device_vector("a_vertex_pair_buffer_src", p_vrtx.data(), p_vrtx.size(), std::cout);
        raft::print_device_vector("a_vertex_pair_buffer_tag", p_tag.data(), p_tag.size(), std::cout);

        
        vertex_frontier.bucket(0).insert(
        thrust::make_zip_iterator(p_vrtx.begin(), p_tag.begin()),
        //thrust::make_zip_iterator(p_vrtx.begin() + 1, p_tag.begin() + 1)
        //thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_p_tag).begin(), std::get<1>(vertex_pair_buffer_p_tag).begin())
        thrust::make_zip_iterator(p_vrtx.end(), p_tag.end())
        );
      } else {
        vertex_frontier.bucket(0).insert(
        thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_p_tag).begin(), std::get<1>(vertex_pair_buffer_p_tag).begin()),
        //thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_p_tag).begin() + 10, std::get<1>(vertex_pair_buffer_p_tag).begin() + 10)
        thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_p_tag).end(), std::get<1>(vertex_pair_buffer_p_tag).end())
        );

      }

  

      rmm::device_uvector<vertex_t> q(0, handle.get_stream());
      rmm::device_uvector<edge_t> idx(0, handle.get_stream());

      //auto [q, idx] =
      auto& comm = handle.get_comms();
      auto comm_rank = comm.get_rank();
      //if (comm_rank == 1) {
      printf("\nbefore calling 'extract_transform_v_frontier_outgoing_e'\n");
      std::tie(q, idx) = 
        cugraph::extract_transform_v_frontier_outgoing_e(
          handle,
          cur_graph_view,
          vertex_frontier.bucket(0),
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          cugraph::edge_dummy_property_t{}.view(),
          extract_q_idx<vertex_t, edge_t>{},
          true);
      
      std::cout << "initial q's size = " << q.size() << std::endl;
      
      raft::print_device_vector("q", q.data(), q.size(), std::cout);
      raft::print_device_vector("i", idx.data(), q.size(), std::cout);
      
      vertex_frontier.bucket(0).clear();

      // Shuffle vertices
      std::tie(q, idx) =
            detail::shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
              handle,
              std::move(q),
              std::move(idx),
              cur_graph_view.vertex_partition_range_lasts());

      vertex_frontier.bucket(0).insert(
      thrust::make_zip_iterator(q.begin(), idx.begin()),
      //thrust::make_zip_iterator(q.begin() + 1, idx.begin() + 1)
      thrust::make_zip_iterator(q.end(), idx.end())
      );
      //}


      auto vertex_pair_buffer_p_r =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(0,
                                                                        handle.get_stream());
      
      auto vertex_pair_buffer_p_q_edge_p_r =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(0,
                                                                        handle.get_stream());
      
      auto vertex_pair_buffer_q_r_edge_p_r =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(0,
                                                                        handle.get_stream());

      // Get chunk global weak edges
      // FIXME: To avoid copying and performing all-to-all
      // when twice when unrolling (q, r) and (p, r) edges, unroll both edge type back to
      // back in with the chunk global weak edgelist

      if constexpr (multi_gpu) {

          // Get global weak edges
          // FIXME: Retrieve onlu a fraction of the weak edges.
        
          auto& comm           = handle.get_comms();
          auto const comm_rank = comm.get_rank();  // FIXME: for debugging
          
          auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
          auto chunk_major_weak_edgelist_srcs = cugraph::detail::device_allgatherv(
            handle, comm, raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()));
          // FIXME: Perform all-to-all in chunks
          auto chunk_major_weak_edgelist_dsts = cugraph::detail::device_allgatherv(
            handle, comm, raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size()));
          
          auto chunk_major_weak_edgelist_tags = cugraph::detail::device_allgatherv(
            handle, comm, raft::device_span<edge_t const>(tag_cpy.data(), tag_cpy.size()));
          
          auto major_weak_edgelist_first = 
            thrust::make_zip_iterator(chunk_major_weak_edgelist_srcs.begin(), chunk_major_weak_edgelist_dsts.begin());

          auto major_weak_edgelist_dsts_tags_first =
            thrust::make_zip_iterator(chunk_major_weak_edgelist_dsts.begin(), chunk_major_weak_edgelist_tags.begin());
          
          thrust::sort_by_key(handle.get_thrust_policy(),
            major_weak_edgelist_dsts_tags_first,
            major_weak_edgelist_dsts_tags_first + chunk_major_weak_edgelist_dsts.size(),
            chunk_major_weak_edgelist_srcs.begin()
            );
          
          // Get global weak_edgelist
          // FIXME: Perform all-to-all in chunks
          auto chunk_global_weak_edgelist_srcs = cugraph::detail::device_allgatherv(
            handle, comm, raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()));
          // FIXME: Perform all-to-all in chunks
          auto chunk_global_weak_edgelist_dsts = cugraph::detail::device_allgatherv(
            handle, comm, raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size()));
          
          raft::print_device_vector("weak_edgelist_tags", tag_cpy.data(), tag_cpy.size(), std::cout);
          auto chunk_global_weak_edgelist_tags = cugraph::detail::device_allgatherv(
            handle, comm, raft::device_span<edge_t const>(tag_cpy.data(), tag_cpy.size()));

          
          
          // Sort the weak edges if they are not already
          auto chunk_global_weak_edgelist_first =
            thrust::make_zip_iterator(chunk_global_weak_edgelist_srcs.begin(), chunk_global_weak_edgelist_dsts.begin());
          /*  
          thrust::sort_by_key(handle.get_thrust_policy(),
                       chunk_global_weak_edgelist_first,
                       chunk_global_weak_edgelist_first + chunk_global_weak_edgelist_srcs.size(),
                       chunk_global_weak_edgelist_tags.begin());
          */
          
          /*
          thrust::sort_by_key(handle.get_thrust_policy(),
                        chunk_global_weak_edgelist_tags.begin(),
                        chunk_global_weak_edgelist_tags.end(),
                        chunk_global_weak_edgelist_first
                        );
          */
          
          
          raft::print_device_vector("chunk_global_weak_edgelist_srcs", chunk_global_weak_edgelist_srcs.data(), chunk_global_weak_edgelist_srcs.size(), std::cout);
          raft::print_device_vector("chunk_global_weak_edgelist_dsts", chunk_global_weak_edgelist_dsts.data(), chunk_global_weak_edgelist_dsts.size(), std::cout);
          raft::print_device_vector("chunk_global_weak_edgelist_tags", chunk_global_weak_edgelist_tags.data(), chunk_global_weak_edgelist_tags.size(), std::cout);

          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          printf("\nnumber of weak edges before 'extract_q_idx_closing' = %d\n", chunk_global_weak_edgelist_dsts.size());
          auto& comm_           = handle.get_comms();
          auto const comm_rank_ = comm.get_rank();  // FIXME: for debugging, remove after

          printf("\nrank %d bucket_size = %d\n", comm_rank_, vertex_frontier.bucket(0).size());

          //#if 0
          // FIXME: Might not even need the 'idx_closing' anymore - remove it
          auto [q_closing, r_closing, p_closing, idx_closing] =
            cugraph::extract_transform_v_frontier_outgoing_e(
              handle,
              cur_graph_view,
              vertex_frontier.bucket(0),
              cugraph::edge_src_dummy_property_t{}.view(),
              cugraph::edge_dst_dummy_property_t{}.view(),
              cugraph::edge_dummy_property_t{}.view(),
              extract_q_idx_closing<vertex_t, edge_t, decltype(chunk_global_weak_edgelist_first), multi_gpu>{
                major_weak_edgelist_dsts_tags_first,
                major_weak_edgelist_dsts_tags_first + chunk_major_weak_edgelist_dsts.size(),
                raft::device_span<vertex_t>(chunk_major_weak_edgelist_srcs.data(), chunk_major_weak_edgelist_srcs.size()),
                raft::device_span<vertex_t>(chunk_global_weak_edgelist_dsts.data(), chunk_global_weak_edgelist_dsts.size()), // FIXME: Unused, remove afterwards
                raft::device_span<edge_t>(chunk_global_weak_edgelist_tags.data(), chunk_global_weak_edgelist_tags.size())
                },
              true);
          
          //#if 0
          raft::print_device_vector("q_closing", q_closing.data(), q_closing.size(), std::cout);
          raft::print_device_vector("idx_closing", idx_closing.data(), idx_closing.size(), std::cout);

          std::cout << "num_closing_edges = " << q_closing.size() << std::endl;

          // extract pair (p, r)
          /*
          auto vertex_pair_buffer_p_r =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(q_closing.size(),
                                                                        handle.get_stream());
          */

          resize_dataframe_buffer(vertex_pair_buffer_p_r,
                                   q_closing.size(),
                                   handle.get_stream());
          /*
          thrust::sort_by_key(handle.get_thrust_policy(),
                        chunk_global_weak_edgelist_tags.begin(),
                        chunk_global_weak_edgelist_tags.end(),
                        chunk_global_weak_edgelist_first
                        );
          */
          thrust::copy(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(p_closing.begin(), r_closing.begin()),
          thrust::make_zip_iterator(p_closing.end(), r_closing.end()),
          thrust::make_zip_iterator(
            std::get<0>(vertex_pair_buffer_p_r).begin(), std::get<1>(vertex_pair_buffer_p_r).begin())
          );
          /*
          auto closing_r_tag = thrust::make_zip_iterator(r_closing.begin(), idx_closing.begin());
          thrust::tabulate(
              handle.get_thrust_policy(),
              get_dataframe_buffer_begin(vertex_pair_buffer_p_r),
              get_dataframe_buffer_end(vertex_pair_buffer_p_r),
              generate_p_r<vertex_t, edge_t, decltype(weak_edgelist_first)>{
                //weak_edgelist_first, // FIXME: might need to use
                //major_weak_edgelist_first,
                //major_weak_edgelist_first + chunk_major_weak_edgelist_tags.size()
                //chunk_global_weak_edgelist_first,
                major_weak_edgelist_first,
                major_weak_edgelist_dsts_tags_first,
                major_weak_edgelist_dsts_tags_first + chunk_major_weak_edgelist_tags.size(),
                closing_r_tag,
                raft::device_span<edge_t const>(idx_closing.data(),
                                                idx_closing.size()),
                raft::device_span<edge_t const>(chunk_global_weak_edgelist_tags.data(), // FIXME: Unused
                                                chunk_global_weak_edgelist_tags.size())
                });
          */
          
          //raft::print_device_vector("check_vertex_pair_buffer_p_r_tags", chunk_major_weak_edgelist_tags.data(), chunk_major_weak_edgelist_tags.size(), std::cout);
          
          //raft::print_device_vector("check_vertex_pair_buffer_p_r_srcs", std::get<0>(vertex_pair_buffer_p_r).data(), std::get<0>(vertex_pair_buffer_p_r).size(), std::cout);
          //raft::print_device_vector("check_vertex_pair_buffer_p_r_dsts", std::get<1>(vertex_pair_buffer_p_r).data(), std::get<1>(vertex_pair_buffer_p_r).size(), std::cout);
          
          
          // construct pair (p, q)
          /*
          auto vertex_pair_buffer_p_q_edge_p_r =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(q_closing.size(),
                                                                        handle.get_stream());
          */

          resize_dataframe_buffer(vertex_pair_buffer_p_q_edge_p_r,
                                   q_closing.size(),
                                   handle.get_stream());
          
          thrust::copy(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(p_closing.begin(), q_closing.begin()),
          thrust::make_zip_iterator(p_closing.end(), q_closing.end()),
          thrust::make_zip_iterator(
            std::get<0>(vertex_pair_buffer_p_q_edge_p_r).begin(), std::get<1>(vertex_pair_buffer_p_q_edge_p_r).begin())
          );

          /*
          thrust::tabulate(
            handle.get_thrust_policy(),
            get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_p_r),
            get_dataframe_buffer_end(vertex_pair_buffer_p_q_edge_p_r),
            generate_p_q_q_r<vertex_t, edge_t, decltype(weak_edgelist_first), true>{
              //major_weak_edgelist_first,
              chunk_global_weak_edgelist_first,
              raft::device_span<vertex_t const>(q_closing.data(),
                                              q_closing.size()),
              raft::device_span<edge_t const>(idx_closing.data(),
                                              idx_closing.size()),
              raft::device_span<edge_t const>(chunk_global_weak_edgelist_tags.data(),
                                              chunk_global_weak_edgelist_tags.size())
              });
          */


          std::cout << "Before remove overcompensating edges when unrolling (p, r) edges" << std::endl;

          raft::print_device_vector("vertex_pair_buffer_p_q_for_p_r_srcs", std::get<0>(vertex_pair_buffer_p_q_edge_p_r).data(), std::get<0>(vertex_pair_buffer_p_q_edge_p_r).size(), std::cout);
          raft::print_device_vector("vertex_pair_buffer_p_q_for_p_r_dsts", std::get<1>(vertex_pair_buffer_p_q_edge_p_r).data(), std::get<1>(vertex_pair_buffer_p_q_edge_p_r).size(), std::cout);

          // construct pair (q, r)
          /*
          auto vertex_pair_buffer_q_r_edge_p_r =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(q_closing.size(),
                                                                        handle.get_stream());
          */

          printf("\nbefore resizing = %d, after resizing = %d\n", size_dataframe_buffer(vertex_pair_buffer_q_r_edge_p_r), q_closing.size()); 

          resize_dataframe_buffer(vertex_pair_buffer_q_r_edge_p_r,
                                   q_closing.size(),
                                   handle.get_stream());
          /*
          printf("\nDone resizing\n");
          thrust::tabulate(
            handle.get_thrust_policy(),
            get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_r),
            get_dataframe_buffer_end(vertex_pair_buffer_q_r_edge_p_r),
            generate_p_q_q_r<vertex_t, edge_t, decltype(weak_edgelist_first), false>{
              chunk_global_weak_edgelist_first,
              raft::device_span<vertex_t const>(q_closing.data(),
                                              q_closing.size()),
              raft::device_span<edge_t const>(idx_closing.data(),
                                              idx_closing.size())
              });
          */

          thrust::copy(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(q_closing.begin(), r_closing.begin()),
          thrust::make_zip_iterator(q_closing.end(), r_closing.end()),
          thrust::make_zip_iterator(
            std::get<0>(vertex_pair_buffer_q_r_edge_p_r).begin(), std::get<1>(vertex_pair_buffer_q_r_edge_p_r).begin())
          );
          
          printf("\nDone generating 'p_q_q_r'\n");
          raft::print_device_vector("vertex_pair_buffer_q_r_for_p_r_srcs", std::get<0>(vertex_pair_buffer_q_r_edge_p_r).data(), std::get<0>(vertex_pair_buffer_q_r_edge_p_r).size(), std::cout);
          raft::print_device_vector("vertex_pair_buffer_q_r_for_p_r_dsts", std::get<1>(vertex_pair_buffer_q_r_edge_p_r).data(), std::get<1>(vertex_pair_buffer_q_r_edge_p_r).size(), std::cout);
        




          std::cout << "num (p, r) edges before removing = " << q_closing.size() << std::endl;
          //raft::print_device_vector("sorted_weak_edgelist_srcs", sorted_weak_edgelist_srcs, weak_edgelist_size, std::cout);
          //raft::print_device_vector("sorted_weak_edgelist_dsts", sorted_weak_edgelist_dsts, weak_edgelist_size, std::cout);

          // FIXME: Check if neccessary
          /*
          auto chunk_global_weak_edgelist_first =
            thrust::make_zip_iterator(chunk_global_weak_edgelist_srcs.begin(), chunk_global_weak_edgelist_dsts.begin());
          */

          // Resort the edges.
          thrust::sort(handle.get_thrust_policy(),
                       chunk_global_weak_edgelist_first,
                       chunk_global_weak_edgelist_first + chunk_global_weak_edgelist_srcs.size());
          
          printf("\nDone sorting\n");
          auto num_edges_not_overcomp_p_q =
            remove_overcompensating_edges<vertex_t,
                                          edge_t,
                                          decltype(get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_p_r)),
                                          false,
                                          false // FIXME: Set it to False for now *******
                                          >(
              handle,
              q_closing.size(),
              get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_p_r), // FIXME: cannot be a copy, needs to be the original one so overcompensatiing edges can be removed
              get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_r), // FIXME: cannot be a copy, needs to be the original one so overcompensatiing edges can be removed
              raft::device_span<vertex_t const>(chunk_global_weak_edgelist_srcs.data(), chunk_global_weak_edgelist_srcs.size()),
              raft::device_span<vertex_t const>(chunk_global_weak_edgelist_dsts.data(), chunk_global_weak_edgelist_dsts.size()),
              //raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
              //raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size()),
              cur_graph_view.vertex_partition_range_lasts()
              );

          std::cout << "1) num (p, r) edges after removing = " << num_edges_not_overcomp_p_q << std::endl;
        
          resize_dataframe_buffer(vertex_pair_buffer_p_q_edge_p_r, num_edges_not_overcomp_p_q, handle.get_stream());
          resize_dataframe_buffer(vertex_pair_buffer_q_r_edge_p_r, num_edges_not_overcomp_p_q, handle.get_stream());

          raft::print_device_vector("vertex_pair_buffer_p_q_for_p_r_srcs", std::get<0>(vertex_pair_buffer_p_q_edge_p_r).data(), std::get<0>(vertex_pair_buffer_p_q_edge_p_r).size(), std::cout);
          raft::print_device_vector("vertex_pair_buffer_p_q_for_p_r_dsts", std::get<1>(vertex_pair_buffer_p_q_edge_p_r).data(), std::get<1>(vertex_pair_buffer_p_q_edge_p_r).size(), std::cout);

          raft::print_device_vector("vertex_pair_buffer_q_r_for_p_r_srcs", std::get<0>(vertex_pair_buffer_q_r_edge_p_r).data(), std::get<0>(vertex_pair_buffer_q_r_edge_p_r).size(), std::cout);
          raft::print_device_vector("vertex_pair_buffer_q_r_for_p_r_dsts", std::get<1>(vertex_pair_buffer_q_r_edge_p_r).data(), std::get<1>(vertex_pair_buffer_q_r_edge_p_r).size(), std::cout);

          //break; // FIXME: Break here **************
          std::cout << "Before remove overcompensating edges when unrolling (p, r) edges" << std::endl;
          raft::print_device_vector("vertex_pair_buffer_p_q_for_p_r_srcs", std::get<0>(vertex_pair_buffer_p_q_edge_p_r).data(), std::get<0>(vertex_pair_buffer_p_q_edge_p_r).size(), std::cout);
          raft::print_device_vector("vertex_pair_buffer_p_q_for_p_r_dsts", std::get<1>(vertex_pair_buffer_p_q_edge_p_r).data(), std::get<1>(vertex_pair_buffer_p_q_edge_p_r).size(), std::cout);

          //break; // FIXME: Break here **************
          auto num_edges_not_overcomp_q_r =
            remove_overcompensating_edges<vertex_t,
                                          edge_t,
                                          decltype(get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_p_r)),
                                          false,
                                          false // FIXME: Set it to False for now
                                          >(
              handle,
              num_edges_not_overcomp_p_q,
              get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_r), // FIXME: cannot be a copy, needs to be the original one so overcompensatiing edges can be removed
              get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_p_r), // FIXME: cannot be a copy, needs to be the original one so overcompensatiing edges can be removed
              raft::device_span<vertex_t const>(chunk_global_weak_edgelist_srcs.data(), chunk_global_weak_edgelist_srcs.size()),
              raft::device_span<vertex_t const>(chunk_global_weak_edgelist_dsts.data(), chunk_global_weak_edgelist_dsts.size()),
              cur_graph_view.vertex_partition_range_lasts());
          
          std::cout << "2) num (p, r) edges after removing = " << num_edges_not_overcomp_q_r << std::endl;
          
          resize_dataframe_buffer(vertex_pair_buffer_q_r_edge_p_r, num_edges_not_overcomp_q_r, handle.get_stream());
          resize_dataframe_buffer(vertex_pair_buffer_p_q_edge_p_r, num_edges_not_overcomp_q_r, handle.get_stream());

          // Reconstruct (p, r) edges that didn't already have their count updated
          // FIXME: No need to reconstruct the third array because we can zip all 3 edges of the triangle
          resize_dataframe_buffer(vertex_pair_buffer_p_r, num_edges_not_overcomp_q_r, handle.get_stream());
            thrust::tabulate(
              handle.get_thrust_policy(),
              get_dataframe_buffer_begin(vertex_pair_buffer_p_r),
              get_dataframe_buffer_end(vertex_pair_buffer_p_r),
              [
                vertex_pair_buffer_p_q_edge_p_r = get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_p_r),
                vertex_pair_buffer_q_r_edge_p_r = get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_r)
              ] __device__(auto i) {
                return thrust::make_tuple(thrust::get<0>(vertex_pair_buffer_p_q_edge_p_r[i]), thrust::get<1>(vertex_pair_buffer_q_r_edge_p_r[i]));
              });
          
          std::cout << "after removing overcompensating edges" << std::endl;
          raft::print_device_vector("check_vertex_pair_buffer_p_r_srcs", std::get<0>(vertex_pair_buffer_p_r).data(), std::get<0>(vertex_pair_buffer_p_r).size(), std::cout);
          raft::print_device_vector("check_vertex_pair_buffer_p_r_dsts", std::get<1>(vertex_pair_buffer_p_r).data(), std::get<1>(vertex_pair_buffer_p_r).size(), std::cout);
          raft::print_device_vector("vertex_pair_buffer_p_q_for_p_r_srcs", std::get<0>(vertex_pair_buffer_p_q_edge_p_r).data(), std::get<0>(vertex_pair_buffer_p_q_edge_p_r).size(), std::cout);
          raft::print_device_vector("vertex_pair_buffer_p_q_for_p_r_dsts", std::get<1>(vertex_pair_buffer_p_q_edge_p_r).data(), std::get<1>(vertex_pair_buffer_p_q_edge_p_r).size(), std::cout);
        //#endif
          
        } else {
          
          // FIXME: refactor SG to use r_closing
          auto [q_closing, r_closing, idx_closing] =
            cugraph::extract_transform_v_frontier_outgoing_e(
              handle,
              cur_graph_view,
              vertex_frontier.bucket(0),
              cugraph::edge_src_dummy_property_t{}.view(),
              cugraph::edge_dst_dummy_property_t{}.view(),
              cugraph::edge_dummy_property_t{}.view(),
              extract_q_idx_closing<vertex_t, edge_t, decltype(weak_edgelist_first), multi_gpu>{raft::device_span<vertex_t>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size())},
              do_expensive_check);
          
          // extract pair (p, r)
          /*
          auto vertex_pair_buffer_p_r =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(q_closing.size(),
                                                                        handle.get_stream());
          */
          resize_dataframe_buffer(vertex_pair_buffer_p_r,
                                  q_closing.size(),
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
          /*
          auto vertex_pair_buffer_p_q_edge_p_r =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(q_closing.size(),
                                                                        handle.get_stream());
          */
          resize_dataframe_buffer(vertex_pair_buffer_p_q_edge_p_r,
                                  q_closing.size(),
                                  handle.get_stream());
          thrust::tabulate(
            handle.get_thrust_policy(),
            get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_p_r),
            get_dataframe_buffer_end(vertex_pair_buffer_p_q_edge_p_r),
            generate_p_q_q_r<vertex_t, edge_t, decltype(weak_edgelist_first), true>{
              weak_edgelist_first,
              raft::device_span<vertex_t const>(q_closing.data(),
                                              q_closing.size()),
              raft::device_span<edge_t const>(idx_closing.data(),
                                              idx_closing.size())
              });
          
          std::cout << "Before remove overcompensating edges when unrolling (p, r) edges" << std::endl;
          raft::print_device_vector("vertex_pair_buffer_p_q_for_p_r_srcs", std::get<0>(vertex_pair_buffer_p_q_edge_p_r).data(), std::get<0>(vertex_pair_buffer_p_q_edge_p_r).size(), std::cout);
          raft::print_device_vector("vertex_pair_buffer_p_q_for_p_r_dsts", std::get<1>(vertex_pair_buffer_p_q_edge_p_r).data(), std::get<1>(vertex_pair_buffer_p_q_edge_p_r).size(), std::cout);

          // construct pair (q, r)
          /*
          auto vertex_pair_buffer_q_r_edge_p_r =
            allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(q_closing.size(),
                                                                        handle.get_stream());
          */
          resize_dataframe_buffer(vertex_pair_buffer_q_r_edge_p_r,
                                  q_closing.size(),
                                  handle.get_stream());
          thrust::tabulate(
            handle.get_thrust_policy(),
            get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_r),
            get_dataframe_buffer_end(vertex_pair_buffer_q_r_edge_p_r),
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
                                          decltype(get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_p_r)),
                                          false,
                                          false // FIXME: Set it to False for now
                                          >(
              handle,
              q_closing.size(),
              get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_p_r), // FIXME: cannot be a copy, needs to be the original one so overcompensatiing edges can be removed
              get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_r), // FIXME: cannot be a copy, needs to be the original one so overcompensatiing edges can be removed
              raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
              raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size()),
              cur_graph_view.vertex_partition_range_lasts());

          resize_dataframe_buffer(vertex_pair_buffer_p_q_edge_p_r, num_edges_not_overcomp_p_q, handle.get_stream());
          resize_dataframe_buffer(vertex_pair_buffer_q_r_edge_p_r, num_edges_not_overcomp_p_q, handle.get_stream());

          std::cout << "Before removing overcompensating edges when unrolling (p, r) edges" << std::endl;
          raft::print_device_vector("vertex_pair_buffer_p_q_for_p_r_srcs", std::get<0>(vertex_pair_buffer_p_q_edge_p_r).data(), std::get<0>(vertex_pair_buffer_p_q_edge_p_r).size(), std::cout);
          raft::print_device_vector("vertex_pair_buffer_p_q_for_p_r_dsts", std::get<1>(vertex_pair_buffer_p_q_edge_p_r).data(), std::get<1>(vertex_pair_buffer_p_q_edge_p_r).size(), std::cout);

          auto num_edges_not_overcomp_q_r =
            remove_overcompensating_edges<vertex_t,
                                          edge_t,
                                          decltype(get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_p_r)),
                                          false,
                                          false // FIXME: Set it to False for now
                                          >(
              handle,
              num_edges_not_overcomp_p_q,
              get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_r), // FIXME: cannot be a copy, needs to be the original one so overcompensatiing edges can be removed
              get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_p_r), // FIXME: cannot be a copy, needs to be the original one so overcompensatiing edges can be removed
              raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
              raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size()),
              cur_graph_view.vertex_partition_range_lasts());
          
          resize_dataframe_buffer(vertex_pair_buffer_p_q_edge_p_r, num_edges_not_overcomp_q_r, handle.get_stream());
          resize_dataframe_buffer(vertex_pair_buffer_q_r_edge_p_r, num_edges_not_overcomp_q_r, handle.get_stream());

          // Reconstruct (p, r) edges that didn't already have their count updated.
          // FIXME: No need to reconstruct the third array because we can zip all 3 edges of the triangle
          resize_dataframe_buffer(vertex_pair_buffer_p_r, num_edges_not_overcomp_q_r, handle.get_stream());
            thrust::tabulate(
              handle.get_thrust_policy(),
              get_dataframe_buffer_begin(vertex_pair_buffer_p_r),
              get_dataframe_buffer_end(vertex_pair_buffer_p_r),
              [
                vertex_pair_buffer_p_q_edge_p_r = get_dataframe_buffer_begin(vertex_pair_buffer_p_q_edge_p_r),
                vertex_pair_buffer_q_r_edge_p_r = get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_r)
              ] __device__(auto i) {
                return thrust::make_tuple(thrust::get<0>(vertex_pair_buffer_p_q_edge_p_r[i]), thrust::get<1>(vertex_pair_buffer_q_r_edge_p_r[i]));
              });

        }
      
      if constexpr (multi_gpu) { 
        // Shuffle before updating count
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_r_srcs(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_r_dsts(0, handle.get_stream());

        std::tie(vertex_pair_buffer_p_r_srcs, vertex_pair_buffer_p_r_dsts, std::ignore, std::ignore, std::ignore, std::ignore) =
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

        std::cout << "updating count_0" <<std::endl;

        raft::print_device_vector("vertex_pair_buffer_p_r_srcs", vertex_pair_buffer_p_r_srcs.data(), vertex_pair_buffer_p_r_srcs.size(), std::cout);
        raft::print_device_vector("vertex_pair_buffer_p_r_dsts", vertex_pair_buffer_p_r_dsts.data(), vertex_pair_buffer_p_r_dsts.size(), std::cout);
        decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
          handle,
          cur_graph_view,
          edge_triangle_counts,
          raft::device_span<vertex_t>(vertex_pair_buffer_p_r_srcs.data(), vertex_pair_buffer_p_r_srcs.size()),
          raft::device_span<vertex_t>(vertex_pair_buffer_p_r_dsts.data(), vertex_pair_buffer_p_r_dsts.size())
        );

        std::cout << "updating count_1" <<std::endl;

        // Shuffle before updating count
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_q_edge_p_r_srcs(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_p_q_edge_p_r_dsts(0, handle.get_stream());
        
        std::tie(vertex_pair_buffer_p_q_edge_p_r_srcs, vertex_pair_buffer_p_q_edge_p_r_dsts, std::ignore, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                      edge_t,
                                                                                      weight_t,
                                                                                      int32_t>(
          handle,
          std::move(std::get<0>(vertex_pair_buffer_p_q_edge_p_r)), // FIXME: rename to vertex_pair_buffer_p_q_edge_p_r for consistency
          std::move(std::get<1>(vertex_pair_buffer_p_q_edge_p_r)), // FIXME: rename to vertex_pair_buffer_p_q_edge_p_r for consistency
          std::nullopt,
          std::nullopt,
          std::nullopt,
          cur_graph_view.vertex_partition_range_lasts());
        
        raft::print_device_vector("vertex_pair_buffer_p_q_edge_p_r_srcs", vertex_pair_buffer_p_q_edge_p_r_srcs.data(), vertex_pair_buffer_p_q_edge_p_r_srcs.size(), std::cout);
        raft::print_device_vector("vertex_pair_buffer_p_q_edge_p_r_dsts", vertex_pair_buffer_p_q_edge_p_r_dsts.data(), vertex_pair_buffer_p_q_edge_p_r_dsts.size(), std::cout);

        decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
          handle,
          cur_graph_view,
          edge_triangle_counts,
          raft::device_span<vertex_t>(vertex_pair_buffer_p_q_edge_p_r_srcs.data(), vertex_pair_buffer_p_q_edge_p_r_srcs.size()),
          raft::device_span<vertex_t>(vertex_pair_buffer_p_q_edge_p_r_dsts.data(), vertex_pair_buffer_p_q_edge_p_r_dsts.size())
        );

        std::cout << "updating count_2" <<std::endl;

        // Shuffle before updating count
        rmm::device_uvector<vertex_t> vertex_pair_buffer_q_r_edge_p_r_srcs(0, handle.get_stream());
        rmm::device_uvector<vertex_t> vertex_pair_buffer_q_r_edge_p_r_dsts(0, handle.get_stream());
        
        std::tie(vertex_pair_buffer_q_r_edge_p_r_srcs, vertex_pair_buffer_q_r_edge_p_r_dsts, std::ignore, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                      edge_t,
                                                                                      weight_t,
                                                                                      int32_t>(
          handle,
          std::move(std::get<0>(vertex_pair_buffer_q_r_edge_p_r)), // FIXME: rename to vertex_pair_buffer_p_q_edge_p_r for consistency
          std::move(std::get<1>(vertex_pair_buffer_q_r_edge_p_r)), // FIXME: rename to vertex_pair_buffer_p_q_edge_p_r for consistency
          std::nullopt,
          std::nullopt,
          std::nullopt,
          cur_graph_view.vertex_partition_range_lasts());
        
        raft::print_device_vector("vertex_pair_buffer_q_r_edge_p_r_srcs", vertex_pair_buffer_q_r_edge_p_r_srcs.data(), vertex_pair_buffer_q_r_edge_p_r_srcs.size(), std::cout);
        raft::print_device_vector("vertex_pair_buffer_q_r_edge_p_r_dsts", vertex_pair_buffer_q_r_edge_p_r_dsts.data(), vertex_pair_buffer_q_r_edge_p_r_dsts.size(), std::cout);

        decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
          handle,
          cur_graph_view,
          edge_triangle_counts,
          raft::device_span<vertex_t>(vertex_pair_buffer_q_r_edge_p_r_srcs.data(), vertex_pair_buffer_q_r_edge_p_r_srcs.size()),
          raft::device_span<vertex_t>(vertex_pair_buffer_q_r_edge_p_r_dsts.data(), vertex_pair_buffer_q_r_edge_p_r_dsts.size())
        );

        std::cout << "Done updating count_2" <<std::endl;
      
      } else {
        decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
          handle,
          cur_graph_view,
          edge_triangle_counts,
          raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_r).data(), std::get<0>(vertex_pair_buffer_p_r).size()),
          raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_p_r).data(), std::get<1>(vertex_pair_buffer_p_r).size())
        );
        
        decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
          handle,
          cur_graph_view,
          edge_triangle_counts,
          raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_p_q_edge_p_r).data(), std::get<0>(vertex_pair_buffer_p_q_edge_p_r).size()),
          raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_p_q_edge_p_r).data(), std::get<1>(vertex_pair_buffer_p_q_edge_p_r).size())
        );
        
        decrease_triangle_count<vertex_t, edge_t, weight_t, multi_gpu>(
          handle,
          cur_graph_view,
          edge_triangle_counts,
          raft::device_span<vertex_t>(std::get<0>(vertex_pair_buffer_q_r_edge_p_r).data(), std::get<0>(vertex_pair_buffer_q_r_edge_p_r).size()),
          raft::device_span<vertex_t>(std::get<1>(vertex_pair_buffer_q_r_edge_p_r).data(), std::get<1>(vertex_pair_buffer_q_r_edge_p_r).size())
        );
      }



      printf("\nafter unrolling (p, r) edges\n");
      auto [srcs_2, dsts_2, count_2] = extract_transform_e(handle,
                                                    cur_graph_view,
                                                    cugraph::edge_src_dummy_property_t{}.view(),
                                                    cugraph::edge_dst_dummy_property_t{}.view(),
                                                    //view_concat(edge_triangle_counts.view(), modified_triangle_count.view()),
                                                    edge_triangle_counts.view(),
                                                    extract_edges_and_triangle_counts<vertex_t, edge_t>{});
      
      raft::print_device_vector("unrolled_srcs_2", srcs_2.data(), srcs_2.size(), std::cout);
      raft::print_device_vector("unrolled_dsts_2", dsts_2.data(), dsts_2.size(), std::cout);
      raft::print_device_vector("unrolled_n_tr_2", count_2.data(), count_2.size(), std::cout);

      // Mask all the edges that have 0 count
      cugraph::transform_e(
          handle,
          cur_graph_view,
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          edge_triangle_counts.view(),
          [] __device__(
            auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto count) {
            return count != 0;
          },
          edge_mask.mutable_view(),
          false);

      cur_graph_view.attach_edge_mask(edge_mask.view());

      auto [srcs_3, dsts_3, count_3] = extract_transform_e(handle,
                                                    cur_graph_view,
                                                    cugraph::edge_src_dummy_property_t{}.view(),
                                                    cugraph::edge_dst_dummy_property_t{}.view(),
                                                    //view_concat(edge_triangle_counts.view(), modified_triangle_count.view()),
                                                    edge_triangle_counts.view(),
                                                    extract_edges_and_triangle_counts<vertex_t, edge_t>{});
      
      
      printf("\nafter removing edges with no count: num_edges = %d\n", srcs_3.size());
      
      raft::print_device_vector("unrolled_srcs_3", srcs_3.data(), srcs_3.size(), std::cout);
      raft::print_device_vector("unrolled_dsts_3", dsts_3.data(), dsts_3.size(), std::cout);
      raft::print_device_vector("unrolled_n_tr_3", count_3.data(), count_3.size(), std::cout);

      /*
      auto [edgelist_srcs_, edgelist_dsts_, edgelist_wgts_, dummy_0, dummy_1] =
      decompress_to_edgelist(
        handle,
        cur_graph_view,
        edge_weight_view ? std::make_optional(*edge_weight_view) : std::nullopt,
        std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
        std::optional<raft::device_span<vertex_t const>>(std::nullopt));
      */
      
      //#endif // While loop
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
