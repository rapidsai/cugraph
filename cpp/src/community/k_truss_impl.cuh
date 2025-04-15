/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include "prims/extract_transform_if_e.cuh"
#include "prims/fill_edge_property.cuh"
#include "prims/per_v_pair_dst_nbr_intersection.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh"
#include "prims/update_edge_src_dst_property.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/collect_comm_wrapper.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/util/integer_utils.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <cuda/std/utility>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugraph {

template <typename edge_t>
struct is_k_or_greater_t {
  edge_t k{};
  __device__ bool operator()(edge_t core_number) const { return core_number >= edge_t{k}; }
};

template <typename vertex_t, typename edge_t>
struct extract_triangles_endpoints {
  size_t chunk_start{};
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  raft::device_span<vertex_t const> weak_srcs{};
  raft::device_span<vertex_t const> weak_dsts{};

  __device__ thrust::tuple<vertex_t, vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = cuda::std::distance(intersection_offsets.begin() + 1, itr);

    auto endpoints = thrust::make_tuple(weak_srcs[chunk_start + idx],  // p
                                        weak_dsts[chunk_start + idx],  // q
                                        intersection_indices[i]        // r
    );

    auto p = weak_srcs[chunk_start + idx];
    auto q = weak_dsts[chunk_start + idx];
    auto r = intersection_indices[i];
    // Re-order the endpoints such that p < q < r in order to identify duplicate triangles
    // which will cause overcompensation. comparing the vertex IDs is cheaper than comparing the
    // degrees (d(p) < d(q) < d(r)) which will be done once in the latter stage to retrieve the
    // direction of the edges once the triplet dependency is broken.
    if (p > q) cuda::std::swap(p, q);
    if (p > r) cuda::std::swap(p, r);
    if (q > r) cuda::std::swap(q, r);

    return thrust::make_tuple(p, q, r);
  }
};

namespace {

template <typename vertex_t>
struct exclude_self_loop_t {
  __device__ cuda::std::optional<thrust::tuple<vertex_t, vertex_t>> operator()(
    vertex_t src,
    vertex_t dst,
    cuda::std::nullopt_t,
    cuda::std::nullopt_t,
    cuda::std::nullopt_t) const
  {
    return src != dst
             ? cuda::std::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
             : cuda::std::nullopt;
  }
};

template <typename vertex_t, typename edge_t>
struct extract_low_to_high_degree_edges_from_endpoints_e_op_t {
  raft::device_span<vertex_t const> srcs{};
  raft::device_span<vertex_t const> dsts{};
  raft::device_span<edge_t const> count{};
  __device__ thrust::tuple<vertex_t, vertex_t, edge_t> operator()(vertex_t src,
                                                                  vertex_t dst,
                                                                  edge_t src_out_degree,
                                                                  edge_t dst_out_degree,
                                                                  cuda::std::nullopt_t) const
  {
    auto itr = thrust::lower_bound(thrust::seq,
                                   thrust::make_zip_iterator(srcs.begin(), dsts.begin()),
                                   thrust::make_zip_iterator(srcs.end(), dsts.end()),
                                   thrust::make_tuple(src, dst));

    auto idx = cuda::std::distance(thrust::make_zip_iterator(srcs.begin(), dsts.begin()), itr);

    if (src_out_degree < dst_out_degree) {
      return thrust::make_tuple(src, dst, count[idx]);
    } else if (dst_out_degree < src_out_degree) {
      return thrust::make_tuple(dst, src, count[idx]);
    } else {  // src_out_degree == dst_out_degree
      if (src < dst /* tie-breaking using vertex ID */) {
        return thrust::make_tuple(src, dst, count[idx]);
      } else {
        return thrust::make_tuple(dst, src, count[idx]);
      }
    }
  }
};

template <typename vertex_t, typename edge_t>
struct extract_low_to_high_degree_edges_from_endpoints_pred_op_t {
  raft::device_span<vertex_t const> srcs{};
  raft::device_span<vertex_t const> dsts{};
  __device__ bool operator()(vertex_t src, vertex_t dst, edge_t, edge_t, cuda::std::nullopt_t) const
  {
    return thrust::binary_search(thrust::seq,
                                 thrust::make_zip_iterator(srcs.begin(), dsts.begin()),
                                 thrust::make_zip_iterator(srcs.end(), dsts.end()),
                                 thrust::make_tuple(src, dst));
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

  // 2. Exclude self-loops and edges that do not belong to (k-1)-core

  auto cur_graph_view          = graph_view;
  auto unmasked_cur_graph_view = cur_graph_view;

  if (unmasked_cur_graph_view.has_edge_mask()) { unmasked_cur_graph_view.clear_edge_mask(); }
  // mask for self-loops and edges not part of k-1 core
  cugraph::edge_property_t<decltype(cur_graph_view), bool> undirected_mask(handle);
  {
    // 2.1 Exclude self-loops

    if (cur_graph_view.count_self_loops(handle) > edge_t{0}) {
      // 2.1. Exclude self-loops

      cugraph::edge_property_t<decltype(cur_graph_view), bool> self_loop_edge_mask(handle,
                                                                                   cur_graph_view);
      cugraph::fill_edge_property(
        handle, unmasked_cur_graph_view, self_loop_edge_mask.mutable_view(), false);

      transform_e(
        handle,
        cur_graph_view,
        edge_src_dummy_property_t{}.view(),
        edge_dst_dummy_property_t{}.view(),
        edge_dummy_property_t{}.view(),
        [] __device__(auto src, auto dst, auto, auto, auto) { return src != dst; },
        self_loop_edge_mask.mutable_view());

      undirected_mask = std::move(self_loop_edge_mask);
      if (cur_graph_view.has_edge_mask()) { cur_graph_view.clear_edge_mask(); }
      cur_graph_view.attach_edge_mask(undirected_mask.view());
    }

    // 2.2 Find (k-1)-core and exclude edges that do not belong to (k-1)-core
    {
      rmm::device_uvector<edge_t> core_numbers(cur_graph_view.number_of_vertices(),
                                               handle.get_stream());
      core_number(handle,
                  cur_graph_view,
                  core_numbers.data(),
                  k_core_degree_type_t::OUT,
                  size_t{2},
                  size_t{2});

      edge_src_property_t<decltype(cur_graph_view), bool> edge_src_in_k_minus_1_cores(
        handle, cur_graph_view);
      edge_dst_property_t<decltype(cur_graph_view), bool> edge_dst_in_k_minus_1_cores(
        handle, cur_graph_view);
      auto in_k_minus_1_core_first =
        thrust::make_transform_iterator(core_numbers.begin(), is_k_or_greater_t<edge_t>{k - 1});
      rmm::device_uvector<bool> in_k_minus_1_core_flags(core_numbers.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   in_k_minus_1_core_first,
                   in_k_minus_1_core_first + core_numbers.size(),
                   in_k_minus_1_core_flags.begin());
      update_edge_src_property(handle,
                               cur_graph_view,
                               in_k_minus_1_core_flags.begin(),
                               edge_src_in_k_minus_1_cores.mutable_view());
      update_edge_dst_property(handle,
                               cur_graph_view,
                               in_k_minus_1_core_flags.begin(),
                               edge_dst_in_k_minus_1_cores.mutable_view());

      cugraph::edge_property_t<decltype(cur_graph_view), bool> in_k_minus_1_core_edge_mask(
        handle, cur_graph_view);
      cugraph::fill_edge_property(
        handle, unmasked_cur_graph_view, in_k_minus_1_core_edge_mask.mutable_view(), false);

      transform_e(
        handle,
        cur_graph_view,
        edge_src_in_k_minus_1_cores.view(),
        edge_dst_in_k_minus_1_cores.view(),
        edge_dummy_property_t{}.view(),
        [] __device__(auto, auto, auto src_in_k_minus_1_core, auto dst_in_k_minus_1_core, auto) {
          return src_in_k_minus_1_core && dst_in_k_minus_1_core;
        },
        in_k_minus_1_core_edge_mask.mutable_view());

      undirected_mask = std::move(in_k_minus_1_core_edge_mask);
      if (cur_graph_view.has_edge_mask()) { cur_graph_view.clear_edge_mask(); }
      cur_graph_view.attach_edge_mask(undirected_mask.view());
    }
  }

  // 3. Keep only the edges from a low-degree vertex to a high-degree vertex.

  edge_src_property_t<decltype(cur_graph_view), edge_t> edge_src_out_degrees(handle,
                                                                             cur_graph_view);
  edge_dst_property_t<decltype(cur_graph_view), edge_t> edge_dst_out_degrees(handle,
                                                                             cur_graph_view);

  cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, bool> dodg_mask(
    handle, cur_graph_view);
  {
    auto out_degrees = cur_graph_view.compute_out_degrees(handle);
    update_edge_src_property(
      handle, cur_graph_view, out_degrees.begin(), edge_src_out_degrees.mutable_view());
    update_edge_dst_property(
      handle, cur_graph_view, out_degrees.begin(), edge_dst_out_degrees.mutable_view());

    cugraph::fill_edge_property(
      handle, unmasked_cur_graph_view, dodg_mask.mutable_view(), bool{false});

    cugraph::transform_e(
      handle,
      cur_graph_view,
      edge_src_out_degrees.view(),
      edge_dst_out_degrees.view(),
      edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_out_degree, auto dst_out_degree, auto) {
        return (src_out_degree < dst_out_degree) ? true
               : ((src_out_degree == dst_out_degree) &&
                  (src < dst) /* tie-breaking using vertex ID */)
                 ? true
                 : false;
      },
      dodg_mask.mutable_view(),
      do_expensive_check);

    if (cur_graph_view.has_edge_mask()) { cur_graph_view.clear_edge_mask(); }
    cur_graph_view.attach_edge_mask(dodg_mask.view());
  }

  // 4. Compute triangle count using nbr_intersection and unroll weak edges

  {
    // Mask self loops and edges not being part of k-1 core
    auto weak_edges_mask = std::move(undirected_mask);

    auto edge_triangle_counts =
      edge_triangle_count<vertex_t, edge_t, multi_gpu>(handle, cur_graph_view, false);

    cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edgelist_weak(handle);
    cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edges_to_decrement_count(handle);
    size_t prev_chunk_size = 0;  // FIXME: Add support for chunking

    while (true) {
      // Extract weak edges
      auto [weak_edgelist_srcs, weak_edgelist_dsts] = extract_transform_if_e(
        handle,
        cur_graph_view,
        edge_src_dummy_property_t{}.view(),
        edge_dst_dummy_property_t{}.view(),
        edge_triangle_counts.view(),
        cuda::proclaim_return_type<thrust::tuple<vertex_t, vertex_t>>(
          [] __device__(vertex_t src, vertex_t dst, auto, auto, auto) {
            return thrust::make_tuple(src, dst);
          }),
        cuda::proclaim_return_type<bool>([k] __device__(auto, auto, auto, auto, edge_t count) {
          return ((count < k - 2) && (count != 0));
        }));

      auto weak_edgelist_first =
        thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin());
      auto weak_edgelist_last =
        thrust::make_zip_iterator(weak_edgelist_srcs.end(), weak_edgelist_dsts.end());

      thrust::sort(handle.get_thrust_policy(), weak_edgelist_first, weak_edgelist_last);

      // Perform nbr_intersection of the weak edges from the undirected
      // graph view
      cur_graph_view.clear_edge_mask();

      // Attach the weak edge mask
      cur_graph_view.attach_edge_mask(weak_edges_mask.view());

      auto [intersection_offsets, intersection_indices] = per_v_pair_dst_nbr_intersection(
        handle, cur_graph_view, weak_edgelist_first, weak_edgelist_last, do_expensive_check);

      // This array stores (p, q, r) which are endpoints for the triangles with weak edges

      auto triangles_endpoints =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t, vertex_t>>(
          intersection_indices.size(), handle.get_stream());

      // Extract endpoints for triangles with weak edges
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(triangles_endpoints),
        get_dataframe_buffer_end(triangles_endpoints),
        extract_triangles_endpoints<vertex_t, edge_t>{
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
          raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size())});

      thrust::sort(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(triangles_endpoints),
                   get_dataframe_buffer_end(triangles_endpoints));

      auto unique_triangle_end = thrust::unique(handle.get_thrust_policy(),
                                                get_dataframe_buffer_begin(triangles_endpoints),
                                                get_dataframe_buffer_end(triangles_endpoints));

      auto num_unique_triangles =
        cuda::std::distance(  // Triangles are represented by their endpoints
          get_dataframe_buffer_begin(triangles_endpoints),
          unique_triangle_end);

      resize_dataframe_buffer(triangles_endpoints, num_unique_triangles, handle.get_stream());

      if constexpr (multi_gpu) {
        auto& comm           = handle.get_comms();
        auto const comm_size = comm.get_size();
        auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
        auto const major_comm_size = major_comm.get_size();
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_size = minor_comm.get_size();

        auto vertex_partition_range_lasts = cur_graph_view.vertex_partition_range_lasts();

        rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
          vertex_partition_range_lasts.size(), handle.get_stream());

        raft::update_device(d_vertex_partition_range_lasts.data(),
                            vertex_partition_range_lasts.data(),
                            vertex_partition_range_lasts.size(),
                            handle.get_stream());

        // Shuffle the edges with respect to the undirected graph view to the GPU
        // owning edge (p, q). Remember that the triplet (p, q, r) is ordered based on the
        // vertex ID and not the degree so (p, q) might not be an edge in the DODG but is
        // surely an edge in the undirected graph
        std::tie(triangles_endpoints, std::ignore) = groupby_gpu_id_and_shuffle_values(
          handle.get_comms(),
          get_dataframe_buffer_begin(triangles_endpoints),
          get_dataframe_buffer_end(triangles_endpoints),

          [key_func =
             cugraph::detail::compute_gpu_id_from_int_edge_endpoints_t<vertex_t>{
               raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                                 d_vertex_partition_range_lasts.size()),
               comm_size,
               major_comm_size,
               minor_comm_size}] __device__(auto val) {
            return key_func(thrust::get<0>(val), thrust::get<1>(val));
          },
          handle.get_stream());

        thrust::sort(handle.get_thrust_policy(),
                     get_dataframe_buffer_begin(triangles_endpoints),
                     get_dataframe_buffer_end(triangles_endpoints));

        unique_triangle_end = thrust::unique(handle.get_thrust_policy(),
                                             get_dataframe_buffer_begin(triangles_endpoints),
                                             get_dataframe_buffer_end(triangles_endpoints));

        num_unique_triangles =
          cuda::std::distance(get_dataframe_buffer_begin(triangles_endpoints), unique_triangle_end);
        resize_dataframe_buffer(triangles_endpoints, num_unique_triangles, handle.get_stream());
      }

      auto edgelist_to_update_count = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        3 * num_unique_triangles, handle.get_stream());

      // The order no longer matters since duplicated triangles have been removed
      // Flatten the endpoints to a list of egdes.
      thrust::transform(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(size_dataframe_buffer(edgelist_to_update_count)),
        get_dataframe_buffer_begin(edgelist_to_update_count),
        [num_unique_triangles,
         triangles_endpoints =
           get_dataframe_buffer_begin(triangles_endpoints)] __device__(auto idx) {
          auto idx_triangle           = idx % num_unique_triangles;
          auto idx_vertex_in_triangle = idx / num_unique_triangles;
          auto triangle               = (triangles_endpoints + idx_triangle).get_iterator_tuple();
          vertex_t src;
          vertex_t dst;

          if (idx_vertex_in_triangle == 0) {
            src = *(thrust::get<0>(triangle));
            dst = *(thrust::get<1>(triangle));
          }

          if (idx_vertex_in_triangle == 1) {
            src = *(thrust::get<0>(triangle));
            dst = *(thrust::get<2>(triangle));
          }

          if (idx_vertex_in_triangle == 2) {
            src = *(thrust::get<1>(triangle));
            dst = *(thrust::get<2>(triangle));
          }

          return thrust::make_tuple(src, dst);
        });

      if constexpr (multi_gpu) {
        std::tie(std::get<0>(edgelist_to_update_count),
                 std::get<1>(edgelist_to_update_count),
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore) =
          detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                         edge_t,
                                                                                         weight_t,
                                                                                         int32_t,
                                                                                         int32_t>(
            handle,
            std::move(std::get<0>(edgelist_to_update_count)),
            std::move(std::get<1>(edgelist_to_update_count)),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            cur_graph_view.vertex_partition_range_lasts());
      }

      thrust::sort(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(edgelist_to_update_count),
                   get_dataframe_buffer_end(edgelist_to_update_count));

      auto unique_pair_count =
        thrust::unique_count(handle.get_thrust_policy(),
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

      std::tie(std::get<0>(vertex_pair_buffer_unique),
               std::get<1>(vertex_pair_buffer_unique),
               decrease_count) =
        extract_transform_if_e(
          handle,
          cur_graph_view,
          edge_src_out_degrees.view(),
          edge_dst_out_degrees.view(),
          edge_dummy_property_t{}.view(),
          extract_low_to_high_degree_edges_from_endpoints_e_op_t<vertex_t, edge_t>{
            raft::device_span<vertex_t const>(std::get<0>(vertex_pair_buffer_unique).data(),
                                              std::get<0>(vertex_pair_buffer_unique).size()),
            raft::device_span<vertex_t const>(std::get<1>(vertex_pair_buffer_unique).data(),
                                              std::get<1>(vertex_pair_buffer_unique).size()),
            raft::device_span<edge_t const>(decrease_count.data(), decrease_count.size())},
          extract_low_to_high_degree_edges_from_endpoints_pred_op_t<vertex_t, edge_t>{
            raft::device_span<vertex_t const>(std::get<0>(vertex_pair_buffer_unique).data(),
                                              std::get<0>(vertex_pair_buffer_unique).size()),
            raft::device_span<vertex_t const>(std::get<1>(vertex_pair_buffer_unique).data(),
                                              std::get<1>(vertex_pair_buffer_unique).size())});

      if constexpr (multi_gpu) {
        auto& comm           = handle.get_comms();
        auto const comm_size = comm.get_size();
        auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
        auto const major_comm_size = major_comm.get_size();
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_size        = minor_comm.get_size();
        auto vertex_partition_range_lasts = cur_graph_view.vertex_partition_range_lasts();

        rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
          vertex_partition_range_lasts.size(), handle.get_stream());
        raft::update_device(d_vertex_partition_range_lasts.data(),
                            vertex_partition_range_lasts.data(),
                            vertex_partition_range_lasts.size(),
                            handle.get_stream());

        std::forward_as_tuple(std::tie(std::get<0>(vertex_pair_buffer_unique),
                                       std::get<1>(vertex_pair_buffer_unique),
                                       decrease_count),
                              std::ignore) =
          groupby_gpu_id_and_shuffle_values(
            handle.get_comms(),
            thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_unique).begin(),
                                      std::get<1>(vertex_pair_buffer_unique).begin(),
                                      decrease_count.begin()),
            thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_unique).end(),
                                      std::get<1>(vertex_pair_buffer_unique).end(),
                                      decrease_count.end()),
            [key_func =
               cugraph::detail::compute_gpu_id_from_int_edge_endpoints_t<vertex_t>{
                 raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                                   d_vertex_partition_range_lasts.size()),
                 comm_size,
                 major_comm_size,
                 minor_comm_size}] __device__(auto val) {
              return key_func(thrust::get<0>(val), thrust::get<1>(val));
            },
            handle.get_stream());
      }

      thrust::sort_by_key(handle.get_thrust_policy(),
                          get_dataframe_buffer_begin(vertex_pair_buffer_unique),
                          get_dataframe_buffer_end(vertex_pair_buffer_unique),
                          decrease_count.begin());

      // Update count of weak edges
      edges_to_decrement_count.clear();

      edges_to_decrement_count.insert(std::get<0>(vertex_pair_buffer_unique).begin(),
                                      std::get<0>(vertex_pair_buffer_unique).end(),
                                      std::get<1>(vertex_pair_buffer_unique).begin());

      cur_graph_view.clear_edge_mask();
      // Check for edge existance on the directed graph view
      cur_graph_view.attach_edge_mask(dodg_mask.view());

      // Update count of weak edges from the DODG view
      cugraph::transform_e(
        handle,
        cur_graph_view,
        edges_to_decrement_count,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        edge_triangle_counts.view(),
        [edge_buffer_first =
           thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_unique).begin(),
                                     std::get<1>(vertex_pair_buffer_unique).begin()),
         edge_buffer_last = thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer_unique).end(),
                                                      std::get<1>(vertex_pair_buffer_unique).end()),
         decrease_count   = raft::device_span<edge_t>(
           decrease_count.data(), decrease_count.size())] __device__(auto src,
                                                                     auto dst,
                                                                     cuda::std::nullopt_t,
                                                                     cuda::std::nullopt_t,
                                                                     edge_t count) {
          auto itr_pair = thrust::lower_bound(
            thrust::seq, edge_buffer_first, edge_buffer_last, thrust::make_tuple(src, dst));
          auto idx_pair = cuda::std::distance(edge_buffer_first, itr_pair);
          count -= decrease_count[idx_pair];

          return count;
        },
        edge_triangle_counts.mutable_view(),
        do_expensive_check);

      edgelist_weak.clear();

      thrust::sort(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin()),
        thrust::make_zip_iterator(weak_edgelist_srcs.end(), weak_edgelist_dsts.end()));

      edgelist_weak.insert(
        weak_edgelist_srcs.begin(), weak_edgelist_srcs.end(), weak_edgelist_dsts.begin());

      // Get undirected graph view
      cur_graph_view.clear_edge_mask();
      cur_graph_view.attach_edge_mask(weak_edges_mask.view());

      auto prev_number_of_edges = cur_graph_view.compute_number_of_edges(handle);

      cugraph::transform_e(
        handle,
        cur_graph_view,
        edgelist_weak,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(
          auto src, auto dst, cuda::std::nullopt_t, cuda::std::nullopt_t, cuda::std::nullopt_t) {
          return false;
        },
        weak_edges_mask.mutable_view(),
        do_expensive_check);

      edgelist_weak.clear();

      // shuffle the edges if multi_gpu
      if constexpr (multi_gpu) {
        std::tie(weak_edgelist_dsts,
                 weak_edgelist_srcs,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore) =
          detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                         edge_t,
                                                                                         weight_t,
                                                                                         int32_t,
                                                                                         int32_t>(
            handle,
            std::move(weak_edgelist_dsts),
            std::move(weak_edgelist_srcs),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            cur_graph_view.vertex_partition_range_lasts());
      }

      thrust::sort(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(weak_edgelist_dsts.begin(), weak_edgelist_srcs.begin()),
        thrust::make_zip_iterator(weak_edgelist_dsts.end(), weak_edgelist_srcs.end()));

      edgelist_weak.insert(
        weak_edgelist_dsts.begin(), weak_edgelist_dsts.end(), weak_edgelist_srcs.begin());

      cugraph::transform_e(
        handle,
        cur_graph_view,
        edgelist_weak,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(
          auto src, auto dst, cuda::std::nullopt_t, cuda::std::nullopt_t, cuda::std::nullopt_t) {
          return false;
        },
        weak_edges_mask.mutable_view(),
        do_expensive_check);

      cur_graph_view.attach_edge_mask(weak_edges_mask.view());

      if (prev_number_of_edges == cur_graph_view.compute_number_of_edges(handle)) { break; }

      cur_graph_view.clear_edge_mask();
      cur_graph_view.attach_edge_mask(dodg_mask.view());
    }

    cur_graph_view.clear_edge_mask();
    cur_graph_view.attach_edge_mask(dodg_mask.view());

    cugraph::transform_e(
      handle,
      cur_graph_view,
      cugraph::edge_src_dummy_property_t{}.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      edge_triangle_counts.view(),
      [] __device__(auto src, auto dst, cuda::std::nullopt_t, cuda::std::nullopt_t, auto count) {
        return count == 0 ? false : true;
      },
      dodg_mask.mutable_view(),
      do_expensive_check);

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
        std::optional<raft::device_span<vertex_t const>>{std::nullopt});

    std::tie(edgelist_srcs,
             edgelist_dsts,
             edgelist_wgts,
             std::ignore,
             std::ignore,
             std::ignore,
             std::ignore) =
      symmetrize_edgelist<vertex_t, edge_t, weight_t, int32_t, int32_t, false, multi_gpu>(
        handle,
        std::move(edgelist_srcs),
        std::move(edgelist_dsts),
        std::move(edgelist_wgts),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        false);

    return std::make_tuple(
      std::move(edgelist_srcs), std::move(edgelist_dsts), std::move(edgelist_wgts));
  }
}
}  // namespace cugraph
