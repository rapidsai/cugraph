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
#include <prims/fill_edge_property.cuh>
#include <prims/reduce_op.cuh>
#include <prims/transform_e.cuh>
#include <prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <thrust/adjacent_difference.h>
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

template <typename vertex_t, typename edge_t, typename EdgeIterator>
struct unroll_edge {
  edge_t num_valid_edges{};
  raft::device_span<edge_t> num_triangles{};  // FIXME: invalid type for unsigned integers
  EdgeIterator edge_unrolled{};
  EdgeIterator transposed_valid_edge_first{};
  EdgeIterator transposed_valid_edge_last{};
  EdgeIterator transposed_invalid_edge_last{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    // edges are sorted with destination as key so reverse the edge when looking it
    auto pair = thrust::make_tuple(thrust::get<1>(*(edge_unrolled + i)),
                                   thrust::get<0>(*(edge_unrolled + i)));
    // Find its position in either partition of the transposed edgelist
    // An edge can be in found in either of the two partitions (valid or invalid)

    auto itr = thrust::lower_bound(thrust::seq, transposed_valid_edge_last, transposed_invalid_edge_last, pair);
    auto idx = thrust::distance(transposed_valid_edge_last, itr) + num_valid_edges;

    if (*itr != pair){
      // The edge must be in the first boundary
      itr = thrust::lower_bound(thrust::seq, transposed_valid_edge_first, transposed_valid_edge_last, pair);
      idx = thrust::distance(transposed_valid_edge_first, itr);
    }
    
    assert(*itr == pair);
    cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(num_triangles[idx]);
    auto r = atomic_counter.fetch_sub(edge_t{1}, cuda::std::memory_order_relaxed);
  }
};

template <typename vertex_t, typename edge_t>
struct unroll_edge_invalid {
  edge_t num_valid_edges{};
  raft::device_span<edge_t> num_triangles{};  // FIXME: invalid type for unsigned integers
  raft::device_span<edge_t> num_triangles_invalid{};

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t idx) const
  {
    
    cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(num_triangles[idx + num_valid_edges]);
    auto r = atomic_counter.fetch_sub(edge_t{num_triangles_invalid[idx]}, cuda::std::memory_order_relaxed);
  }
};

template <typename vertex_t, typename edge_t>
rmm::device_uvector<vertex_t> prefix_sum_valid_and_invalid_edges(
  // The edgelist is segmented into 2 partitions (valid and invalid edges)
  // and edges to be unrolled can be either in the valid or the invalid edge
  // partition.
  raft::handle_t const& handle,
  edge_t num_edges,
  edge_t num_invalid_edges,
  vertex_t* invalid_dst,
  raft::device_span<vertex_t> edgelist_dsts)
{
  rmm::device_uvector<vertex_t> prefix_sum(num_invalid_edges + 1, handle.get_stream());

  thrust::tabulate(
    handle.get_thrust_policy(),
    prefix_sum.begin(),
    prefix_sum.begin() + num_invalid_edges, 
    [num_edges,
     invalid_dst,
     edgelist_dsts = edgelist_dsts.begin()] __device__(auto idx) {
      auto itr_lower_valid     = thrust::lower_bound(thrust::seq, edgelist_dsts, edgelist_dsts + num_edges, invalid_dst[idx]);
      auto itr_upper_valid     = thrust::upper_bound(thrust::seq, itr_lower_valid, edgelist_dsts + num_edges, invalid_dst[idx]);
      auto dist_valid          = thrust::distance(itr_lower_valid, itr_upper_valid);

      return dist_valid;
    });
  thrust::exclusive_scan(
    handle.get_thrust_policy(), prefix_sum.begin(), prefix_sum.end(), prefix_sum.begin());

  return prefix_sum;
  
}
 
template <typename vertex_t, typename edge_t, typename EdgeIterator>
edge_t remove_overcompensating_edges(
  raft::handle_t const& handle,
  edge_t num_valid_edges,
  edge_t num_invalid_edges,
  edge_t dataframe_buffer_size,
  EdgeIterator&& potential_closing_or_incoming_edges,
  EdgeIterator&& incoming_or_potential_closing_edges, 
  raft::device_span<vertex_t> edgelist_srcs,
  raft::device_span<vertex_t> edgelist_dsts
  )
{
  // To avoid over-compensating, check whether the 'potential_closing_edges'
  // are within the invalid edges. If yes, the was already unrolled
  auto edges_not_overcomp = thrust::remove_if(handle.get_thrust_policy(),
                                            thrust::make_zip_iterator(get_dataframe_buffer_begin(potential_closing_or_incoming_edges), get_dataframe_buffer_begin(incoming_or_potential_closing_edges)),
                                            thrust::make_zip_iterator(get_dataframe_buffer_begin(potential_closing_or_incoming_edges) + dataframe_buffer_size, get_dataframe_buffer_begin(incoming_or_potential_closing_edges) + dataframe_buffer_size),
                                             [
                                              num_invalid_edges,
                                              num_valid_edges,                                           
                                              invalid_first  = thrust::make_zip_iterator(edgelist_dsts.begin() + num_valid_edges, edgelist_srcs.begin() + num_valid_edges),
                                              invalid_last   = thrust::make_zip_iterator(edgelist_dsts.end(), edgelist_srcs.end())] __device__(auto e) {
                                                auto potential_edge = thrust::get<0>(e);
                                                auto transposed_potential_or_incoming_edge = thrust::make_tuple(thrust::get<1>(potential_edge), thrust::get<0>(potential_edge));
                                                auto itr = thrust::lower_bound(thrust::seq, invalid_first, invalid_last, transposed_potential_or_incoming_edge); //very important when using lower or upperbound on edges that do not exist. Always make sure to compare with the queried edges
                                                assert(*itr == transposed_potential_or_incoming_edge);
                                                auto dist = thrust::distance(invalid_first, itr);
                                                if (*itr != transposed_potential_or_incoming_edge){
                                                  return false;
                                                }
                                                return dist < num_invalid_edges;
                                            });

  auto dist = thrust::distance(
      thrust::make_zip_iterator(get_dataframe_buffer_begin(potential_closing_or_incoming_edges), get_dataframe_buffer_begin(incoming_or_potential_closing_edges)),
      edges_not_overcomp);

  // After pushing the non-existant edges to the second partition,
  // remove them by resizing  both vertex pair buffer
  resize_dataframe_buffer(potential_closing_or_incoming_edges, dist, handle.get_stream());
  resize_dataframe_buffer(incoming_or_potential_closing_edges, dist, handle.get_stream());

  return dist;
}

template <typename vertex_t,
          typename edge_t,
          bool multi_gpu,
          bool is_edge_q_r>
void find_unroll_p_r_and_q_r_edges(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, false>& graph_view,
  edge_t num_invalid_edges,
  edge_t num_valid_edges,
  rmm::device_uvector<vertex_t>&& edgelist_srcs, // FIXME: Use device_span instead
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  rmm::device_uvector<edge_t>&& num_triangles)
{
   auto prefix_sum_valid = prefix_sum_valid_and_invalid_edges(
    handle,
    edge_t{num_valid_edges},
    edge_t{num_invalid_edges}, // num_edges == num_invalid_edges
    edgelist_dsts.begin() + num_valid_edges,
    raft::device_span<vertex_t>(edgelist_dsts.data(), num_valid_edges));

  auto prefix_sum_invalid = prefix_sum_valid_and_invalid_edges(
    handle,
    edge_t{num_invalid_edges},
    edge_t{num_invalid_edges},
    edgelist_dsts.begin() + num_valid_edges,
    raft::device_span<vertex_t>(edgelist_dsts.data() + num_valid_edges, num_invalid_edges));


  auto potential_closing_edges = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
    prefix_sum_valid.back_element(handle.get_stream()) + prefix_sum_invalid.back_element(handle.get_stream()), handle.get_stream());

  auto incoming_edges_to_r = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
    prefix_sum_valid.back_element(handle.get_stream()) + prefix_sum_invalid.back_element(handle.get_stream()), handle.get_stream());

  const bool const_is_edge_q_r = is_edge_q_r;
  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<edge_t>(0),
    thrust::make_counting_iterator<edge_t>(num_invalid_edges),
    [num_valid_edges,
     num_invalid_edges,
     invalid_first_dst       = edgelist_dsts.begin() + num_valid_edges,
     invalid_first_src       = edgelist_srcs.begin() + num_valid_edges,
     src_array_begin_valid   = edgelist_srcs.begin(),
     dst_array_begin_valid   = edgelist_dsts.begin(),
     prefix_sum_valid        = prefix_sum_valid.data(),
     prefix_sum_invalid      = prefix_sum_invalid.data(),
     potential_closing_edges = get_dataframe_buffer_begin(potential_closing_edges),
     incoming_edges_to_r     = get_dataframe_buffer_begin(incoming_edges_to_r),
     const_is_edge_q_r] __device__(auto idx) {
      auto src           = invalid_first_src[idx];
      auto dst           = invalid_first_dst[idx];
      auto dst_array_end_valid = dst_array_begin_valid + num_valid_edges;

      auto itr_lower_valid = thrust::lower_bound(thrust::seq, dst_array_begin_valid, dst_array_end_valid, dst);
      auto idx_lower_valid = thrust::distance(
        dst_array_begin_valid, itr_lower_valid);  // Need a binary search to find the begining of the range
      
      auto invalid_end_dst = invalid_first_dst + num_invalid_edges;
      // FIXME: In case of wrong results, investigate lower bound when dst is not part of one partition
      auto itr_lower_invalid = thrust::lower_bound(thrust::seq, invalid_first_dst, invalid_end_dst, dst);
      auto idx_lower_invalid = thrust::distance(
        invalid_first_dst, itr_lower_invalid);  // Need a binary search to find the begining of the range

      auto incoming_edges_to_r_first_valid =
        thrust::make_zip_iterator(src_array_begin_valid + idx_lower_valid, thrust::make_constant_iterator(dst));
      thrust::copy(thrust::seq,
                   incoming_edges_to_r_first_valid,
                   incoming_edges_to_r_first_valid + (prefix_sum_valid[idx + 1] - prefix_sum_valid[idx]),
                   incoming_edges_to_r + prefix_sum_valid[idx] + prefix_sum_invalid[idx]);
          
      auto incoming_edges_to_r_first_invalid =
        thrust::make_zip_iterator(invalid_first_src  + idx_lower_invalid, thrust::make_constant_iterator(dst));
      thrust::copy(thrust::seq,
                   incoming_edges_to_r_first_invalid,
                   incoming_edges_to_r_first_invalid + (prefix_sum_invalid[idx + 1] - prefix_sum_invalid[idx]),
                   // FIXME remove prefix_sum_valid[idx] as it is substracted
                   incoming_edges_to_r + prefix_sum_invalid[idx] + prefix_sum_valid[idx + 1]);

      if constexpr (const_is_edge_q_r) {
        auto potential_closing_edges_first_valid = thrust::make_zip_iterator(
          src_array_begin_valid + idx_lower_valid, thrust::make_constant_iterator(src));
        thrust::copy(thrust::seq,
                     potential_closing_edges_first_valid,
                     potential_closing_edges_first_valid + (prefix_sum_valid[idx + 1] - prefix_sum_valid[idx]),
                     potential_closing_edges + prefix_sum_valid[idx] + prefix_sum_invalid[idx]);
        
        auto potential_closing_edges_first_invalid = thrust::make_zip_iterator(
          invalid_first_src  + idx_lower_invalid, thrust::make_constant_iterator(src));
        thrust::copy(thrust::seq,
                     potential_closing_edges_first_invalid,
                     potential_closing_edges_first_invalid + (prefix_sum_invalid[idx + 1] - prefix_sum_invalid[idx]),
                     potential_closing_edges + prefix_sum_invalid[idx] + prefix_sum_valid[idx + 1]);

      } else {
        auto potential_closing_edges_first_valid = thrust::make_zip_iterator(
          thrust::make_constant_iterator(src), src_array_begin_valid + idx_lower_valid);
        thrust::copy(thrust::seq,
                     potential_closing_edges_first_valid,
                     potential_closing_edges_first_valid + (prefix_sum_valid[idx + 1] - prefix_sum_valid[idx]),
                     potential_closing_edges + prefix_sum_valid[idx] + prefix_sum_invalid[idx]);
        
        auto potential_closing_edges_first_invalid = thrust::make_zip_iterator(
          thrust::make_constant_iterator(src), invalid_first_src + idx_lower_invalid);
        thrust::copy(thrust::seq,
                     potential_closing_edges_first_invalid,
                     potential_closing_edges_first_invalid + (prefix_sum_invalid[idx + 1] - prefix_sum_invalid[idx]),
                     potential_closing_edges + prefix_sum_invalid[idx] +(prefix_sum_valid[idx + 1]));
      }
    
    });

  auto edges_exist = graph_view.has_edge(
    handle,
    raft::device_span<vertex_t const>(std::get<0>(potential_closing_edges).data(),
                                      std::get<0>(potential_closing_edges).size()),
    raft::device_span<vertex_t const>(std::get<1>(potential_closing_edges).data(),
                                      std::get<1>(potential_closing_edges).size()));

  auto edge_to_existance = thrust::make_zip_iterator(
    thrust::make_zip_iterator(get_dataframe_buffer_begin(potential_closing_edges),
                              get_dataframe_buffer_begin(incoming_edges_to_r)),
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
  resize_dataframe_buffer(potential_closing_edges, num_edge_exists, handle.get_stream());
  resize_dataframe_buffer(incoming_edges_to_r, num_edge_exists, handle.get_stream());

  edge_t num_edges_not_overcomp = remove_overcompensating_edges<vertex_t, edge_t, decltype(potential_closing_edges)>(
    handle,
    edge_t{num_valid_edges},
    edge_t{num_invalid_edges},
    edge_t{num_edge_exists},
    std::move(potential_closing_edges),
    std::move(incoming_edges_to_r),
    raft::device_span<vertex_t>(edgelist_srcs.data(), edgelist_srcs.size()),
    raft::device_span<vertex_t>(edgelist_dsts.data(), edgelist_dsts.size())
  );

  // Extra check for 'incoming_edges_to_r'
  if constexpr (!is_edge_q_r) {
    // Exchange the arguments (incoming_edges_to_r, num_edges_not_overcomp) order
    // To also check if the 'incoming_edges_to_r' belong the the invalid_edgelist
    num_edges_not_overcomp = remove_overcompensating_edges<vertex_t, edge_t, decltype(potential_closing_edges)>(
      handle,
      edge_t{num_valid_edges},
      edge_t{num_invalid_edges},
      edge_t{num_edges_not_overcomp},
      std::move(incoming_edges_to_r),
      std::move(potential_closing_edges),
      raft::device_span<vertex_t>(edgelist_srcs.data(), edgelist_srcs.size()),
      raft::device_span<vertex_t>(edgelist_dsts.data(), edgelist_dsts.size())
    );
  }

  // FIXME: Combine both 'thrust::for_each'
  if constexpr (is_edge_q_r) {
    thrust::for_each(handle.get_thrust_policy(),
                      thrust::make_zip_iterator(get_dataframe_buffer_begin(potential_closing_edges), get_dataframe_buffer_begin(incoming_edges_to_r)),
                      thrust::make_zip_iterator(get_dataframe_buffer_begin(potential_closing_edges) + num_edges_not_overcomp, get_dataframe_buffer_begin(incoming_edges_to_r) + num_edges_not_overcomp),
                      [num_triangles = num_triangles.begin(),
                        num_valid_edges,
                        invalid_first  = thrust::make_zip_iterator(edgelist_dsts.begin() + num_valid_edges, edgelist_srcs.begin() + num_valid_edges),
                        invalid_last   = thrust::make_zip_iterator(edgelist_dsts.end(), edgelist_srcs.end())
                        ] __device__(auto potential_or_incoming_e){
                          auto potential_e = thrust::get<0>(potential_or_incoming_e);
                          auto incoming_e_to_r = thrust::get<1>(potential_or_incoming_e);
                          auto transposed_invalid_edge = thrust::make_tuple(thrust::get<1>(incoming_e_to_r), thrust::get<1>(potential_e));
                          auto itr = thrust::lower_bound(thrust::seq, invalid_first, invalid_last, transposed_invalid_edge);
                          assert(*itr == transposed_invalid_edge);
                          auto dist = thrust::distance(invalid_first, itr) + num_valid_edges;
                          
                          cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(num_triangles[dist]);
                          auto r = atomic_counter.fetch_sub(edge_t{1}, cuda::std::memory_order_relaxed);   
                      });
  }
  else{
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_zip_iterator(get_dataframe_buffer_begin(potential_closing_edges), get_dataframe_buffer_begin(incoming_edges_to_r)),
                     thrust::make_zip_iterator(get_dataframe_buffer_begin(potential_closing_edges) + num_edges_not_overcomp, get_dataframe_buffer_begin(incoming_edges_to_r) + num_edges_not_overcomp),
                     [num_triangles = num_triangles.begin(),
                      num_valid_edges,
                      invalid_first  = thrust::make_zip_iterator(edgelist_dsts.begin() + num_valid_edges, edgelist_srcs.begin() + num_valid_edges),
                      invalid_last   = thrust::make_zip_iterator(edgelist_dsts.end(), edgelist_srcs.end())
                      ] __device__(auto potential_or_incoming_e){
                        auto potential_e = thrust::get<0>(potential_or_incoming_e);
                        auto incoming_e_to_r = thrust::get<1>(potential_or_incoming_e);
                        auto transposed_invalid_edge = thrust::make_tuple(thrust::get<1>(incoming_e_to_r), thrust::get<0>(potential_e));
                        auto itr = thrust::lower_bound(thrust::seq, invalid_first, invalid_last, transposed_invalid_edge);
                        assert(*itr == transposed_invalid_edge);
                        auto dist = thrust::distance(invalid_first, itr) + num_valid_edges;

                        cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(num_triangles[dist]);
                        auto r = atomic_counter.fetch_sub(edge_t{1}, cuda::std::memory_order_relaxed);                          
                     });

  }

  thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_counting_iterator<edge_t>(0),
                   thrust::make_counting_iterator<edge_t>(num_edges_not_overcomp),
                   unroll_edge<vertex_t, edge_t, decltype(thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin()))>{
                     edge_t{num_valid_edges},
                     raft::device_span<edge_t>(num_triangles.data(), num_triangles.size()),
                     get_dataframe_buffer_begin(potential_closing_edges),
                     thrust::make_zip_iterator(edgelist_dsts.begin(), edgelist_srcs.begin()),
                     thrust::make_zip_iterator(edgelist_dsts.begin() + num_valid_edges, edgelist_srcs.begin() + num_valid_edges),
                     thrust::make_zip_iterator(edgelist_dsts.end(), edgelist_srcs.end())});
  
  thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_counting_iterator<edge_t>(0),
                   thrust::make_counting_iterator<edge_t>(num_edges_not_overcomp),
                   unroll_edge<vertex_t, edge_t, decltype(thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin()))>{
                     edge_t{num_valid_edges},
                     raft::device_span<edge_t>(num_triangles.data(), num_triangles.size()),
                     get_dataframe_buffer_begin(incoming_edges_to_r),
                     thrust::make_zip_iterator(edgelist_dsts.begin(), edgelist_srcs.begin()),
                     thrust::make_zip_iterator(edgelist_dsts.begin() + num_valid_edges, edgelist_srcs.begin() + num_valid_edges),
                     thrust::make_zip_iterator(edgelist_dsts.end(), edgelist_srcs.end())});

}

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
struct generate_p_r_q_r {
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  raft::device_span<vertex_t const> invalid_srcs{};
  raft::device_span<vertex_t const> invalid_dsts{};



  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);

    if constexpr (generate_p_r) {  
      return thrust::make_tuple(invalid_srcs[idx],
                                intersection_indices[i]);
      
    } else {
      return thrust::make_tuple(invalid_dsts[idx],
                                intersection_indices[i]);
      
    }
  }
};
}  // namespace

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> k_truss(
  raft::handle_t const& handle,
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

  // 3. Find (k+1)-core and exclude edges that do not belong to (k+1)-core

  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;
    auto vertex_partition_range_lasts =
      renumber_map
        ? std::make_optional<std::vector<vertex_t>>(cur_graph_view.vertex_partition_range_lasts())
        : std::nullopt;

    rmm::device_uvector<edge_t> d_core_numbers(cur_graph_view.local_vertex_partition_range_size(),
                                               handle.get_stream());
    raft::device_span<edge_t const> core_number_span{d_core_numbers.data(), d_core_numbers.size()};

    rmm::device_uvector<vertex_t> srcs{0, handle.get_stream()};
    rmm::device_uvector<vertex_t> dsts{0, handle.get_stream()};
    std::tie(srcs, dsts, std::ignore) =
      k_core(handle,
             cur_graph_view,
             std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
             size_t{k+1},
             std::make_optional(k_core_degree_type_t::OUT),
             // Seems like the below argument is required. passing a std::nullopt
             // create a compiler error
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
        false);  //******************FIXME:        hardcoded to False

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
    
    auto num_triangles = edge_triangle_count<vertex_t, edge_t, false, false>(
      handle,
      cur_graph_view,
      raft::device_span<vertex_t>(edgelist_srcs.data(), edgelist_srcs.size()),
      raft::device_span<vertex_t>(edgelist_dsts.data(), edgelist_dsts.size()));

    // FIXME 'edge_triangle_count' sorts the edges by 'src' but 'k-truss' needs
    // the edges to be sorted with 'dst' as key so we need to sort the edges 
    // again. Should 'edge_triangle_count' be implemented edges sorted by 'dst'
    // instead to avoid resorting?
    auto transposed_edge_first =
      thrust::make_zip_iterator(edgelist_dsts.begin(), edgelist_srcs.begin());
    
    auto transposed_edge_triangle_count_pair_first =
      thrust::make_zip_iterator(transposed_edge_first, num_triangles.begin());

    // Note: ensure 'edges_with_triangles' and 'cur_graph_view' have the same transpose flag
    cugraph::edge_property_t<decltype(cur_graph_view), bool> edge_mask(handle, cur_graph_view);

    edge_t num_invalid_edges{0};
    size_t num_edges_with_triangles{0};

    while (true) {
      // Remove edges that have a triangle count of zero. Those should not be accounted
      // for during the unroling phase.
      auto edges_with_triangle_last =
        thrust::stable_partition(handle.get_thrust_policy(),
                                 transposed_edge_triangle_count_pair_first,
                                 transposed_edge_triangle_count_pair_first + num_triangles.size(),
                                 [k] __device__(auto e) {
                                   auto num_triangles = thrust::get<1>(e);
                                   return num_triangles > 0;
                                 });
      
      /*
      //FIXME: Getting different results than the above
      auto edges_with_triangle_last =
        thrust::remove_if(handle.get_thrust_policy(),
                          transposed_edge_triangle_count_pair_first,
                          transposed_edge_triangle_count_pair_first + num_triangles.size(),
                          [] __device__(auto e) {
                            auto num_triangles = thrust::get<1>(e);
                            return num_triangles == 0;
                          });
      */      

      num_edges_with_triangles = static_cast<size_t>(
        thrust::distance(transposed_edge_triangle_count_pair_first, edges_with_triangle_last));

      cugraph::edge_property_t<decltype(cur_graph_view), bool> edge_mask(handle, cur_graph_view);
      // Set edge property to 'True' for all edges then mask out invalid edges which can be significantly
      // smaller than the valid edges
      cugraph::fill_edge_property(handle, cur_graph_view, true, edge_mask);

      thrust::sort(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(edgelist_srcs.begin() + num_edges_with_triangles, edgelist_dsts.begin() + num_edges_with_triangles),
          thrust::make_zip_iterator(edgelist_srcs.end(), edgelist_dsts.end()));

      cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edges_with_no_triangle(handle);
      edges_with_no_triangle.insert(
        edgelist_srcs.begin() + num_edges_with_triangles, edgelist_srcs.end(),edgelist_dsts.begin() + num_edges_with_triangles);
      
      // FIXME: Cannot modify an edgemask that is still attached.
      // This can lead to race conditions
      cugraph::transform_e(
        handle,
        cur_graph_view,
        edges_with_no_triangle,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {
          return false;
        },
        edge_mask.mutable_view(),
        false); // FIXME: Remove expensive check. This is only here for debugging purposes **********************************

      cur_graph_view.attach_edge_mask(edge_mask.view());

      edgelist_srcs.resize(num_edges_with_triangles, handle.get_stream());
      edgelist_dsts.resize(num_edges_with_triangles, handle.get_stream());
      num_triangles.resize(num_edges_with_triangles, handle.get_stream());
      
      thrust::sort_by_key(
        handle.get_thrust_policy(), transposed_edge_first, transposed_edge_first + edgelist_srcs.size(), num_triangles.begin());

      // 'invalid_transposed_edge_triangle_count_first' marks the beginning of the edges to be removed
      // 'invalid_transposed_edge_triangle_count_first' + edgelist_srcs.size() marks the end of the edges to be removed
      // 'edge_triangle_count_pair_first' marks the begining of the valid edges
      auto invalid_transposed_edge_triangle_count_first =
        thrust::stable_partition(handle.get_thrust_policy(),
                                 transposed_edge_triangle_count_pair_first,
                                 transposed_edge_triangle_count_pair_first + num_triangles.size(),
                                 [k] __device__(auto e) {
                                   auto num_triangles = thrust::get<1>(e);
                                   return num_triangles >= k - 2;
                                 });
      
      num_invalid_edges = static_cast<size_t>(thrust::distance(
        invalid_transposed_edge_triangle_count_first, transposed_edge_triangle_count_pair_first + edgelist_srcs.size()));

      if (num_invalid_edges == 0) { break; }

      auto num_valid_edges = edgelist_srcs.size() - num_invalid_edges;   

      // case 3. For the (p, q), find intersection 'r' to create (p, r, -1) and (q, r, -1)
      // FIXME: check if 'invalid_transposed_edge_triangle_count_first' is necessery as I operate on 'vertex_pair_buffer'
      // which contains the ordering with the number of triangles.
      // FIXME: debug this stage. There are edges that have been removed that are still found in nbr
      // intersection
      auto [intersection_offsets, intersection_indices] =
        detail::nbr_intersection(handle,
                                 cur_graph_view,
                                 cugraph::edge_dummy_property_t{}.view(),
                                 thrust::make_zip_iterator(edgelist_srcs.begin() + num_valid_edges, edgelist_dsts.begin() + num_valid_edges),
                                 thrust::make_zip_iterator(edgelist_srcs.end(), edgelist_dsts.end()),
                                 std::array<bool, 2>{true, true},
                                 do_expensive_check);

      rmm::device_uvector<edge_t> num_triangles_invalid(num_invalid_edges, handle.get_stream());

      // Update the number of triangles of each (p, q) edges by looking at their intersection
      // size
      thrust::adjacent_difference(handle.get_thrust_policy(),
                                  intersection_offsets.begin() + 1,
                                  intersection_offsets.end(),
                                  num_triangles_invalid.begin());
  
      thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_counting_iterator<edge_t>(0),
                   thrust::make_counting_iterator<edge_t>(num_invalid_edges),
                   unroll_edge_invalid<vertex_t, edge_t>{
                     edge_t{num_valid_edges},
                     raft::device_span<edge_t>(num_triangles.data(), num_triangles.size()),
                     raft::device_span<edge_t>(num_triangles_invalid.data(), num_triangles_invalid.size())});

      size_t accumulate_pair_size = intersection_indices.size();

      // FIXME: Find a way to not have to maintain a dataframe_buffer
      auto vertex_pair_buffer_p_r_edge_p_q =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(accumulate_pair_size,
                                                                     handle.get_stream());
    
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_p_q),
        get_dataframe_buffer_end(vertex_pair_buffer_p_r_edge_p_q),
        generate_p_r_q_r<vertex_t,edge_t, true>{
          raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          raft::device_span<vertex_t>(edgelist_srcs.data() + num_valid_edges, num_invalid_edges),
          raft::device_span<vertex_t>(edgelist_dsts.data() + num_valid_edges, num_invalid_edges)});

      auto num_edge_exists = accumulate_pair_size;
      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_counting_iterator<edge_t>(0),
                       thrust::make_counting_iterator<edge_t>(num_edge_exists),
                       unroll_edge<vertex_t, edge_t, decltype(transposed_edge_first)>{
                         edge_t{num_valid_edges},
                         raft::device_span<edge_t>(num_triangles.data(), num_triangles.size()),
                         get_dataframe_buffer_begin(vertex_pair_buffer_p_r_edge_p_q),
                         transposed_edge_first,
                         transposed_edge_first + num_valid_edges,
                         transposed_edge_first + edgelist_srcs.size()
                       });
      
      auto vertex_pair_buffer_q_r_edge_p_q =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(accumulate_pair_size,
                                                                     handle.get_stream());

      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q),
        get_dataframe_buffer_end(vertex_pair_buffer_q_r_edge_p_q),
        generate_p_r_q_r<vertex_t,edge_t, false>{
          raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          raft::device_span<vertex_t>(edgelist_srcs.data() + num_valid_edges, num_invalid_edges),
          raft::device_span<vertex_t>(edgelist_dsts.data() + num_valid_edges, num_invalid_edges)});

      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_counting_iterator<edge_t>(0),
                       thrust::make_counting_iterator<edge_t>(num_edge_exists),
                       unroll_edge<vertex_t, edge_t, decltype(transposed_edge_first)>{
                         edge_t{num_valid_edges},
                         raft::device_span<edge_t>(num_triangles.data(), num_triangles.size()),
                         get_dataframe_buffer_begin(vertex_pair_buffer_q_r_edge_p_q),
                         transposed_edge_first,
                         transposed_edge_first + num_valid_edges,
                         transposed_edge_first + edgelist_srcs.size()
                       });
      
      // case 1: unroll (q, r)
      // For each (q, r) edges to unroll, find the incoming edges to 'r' let's say from 'p' and
      // create the pair (p, q)
      cugraph::find_unroll_p_r_and_q_r_edges<vertex_t,
                                             edge_t,
                                             false,
                                             true>(handle,
                                                    cur_graph_view,
                                                    num_invalid_edges,
                                                    num_valid_edges,
                                                    std::move(edgelist_srcs),
                                                    std::move(edgelist_dsts),
                                                    std::move(num_triangles));
  
      // case 2: unroll (p, r)
      cugraph::find_unroll_p_r_and_q_r_edges<vertex_t,
                                             edge_t,
                                             false,
                                             false>(handle,
                                                    cur_graph_view,
                                                    num_invalid_edges,
                                                    num_valid_edges,
                                                    std::move(edgelist_srcs),
                                                    std::move(edgelist_dsts),
                                                    std::move(num_triangles));
    
      cur_graph_view.clear_edge_mask();
      edges_with_no_triangle.clear();
      edge_mask.clear(handle); // masking not working in a loop.

    }

    printf("\n*********final results*********\n");
    //raft::print_device_vector("srcs", edgelist_srcs.data(), edgelist_srcs.size(), std::cout);
    //raft::print_device_vector("dsts", edgelist_dsts.data(), edgelist_dsts.size(), std::cout);
    //raft::print_device_vector("n_tr", num_triangles.data(), num_triangles.size(), std::cout);
    printf("\nthe number of edges = %d\n", edgelist_srcs.size());
    return std::make_tuple(std::move(edgelist_srcs), std::move(edgelist_dsts));
  }
}
}  // namespace cugraph