/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "prims/extract_transform_if_e.cuh"
#include "prims/fill_edge_property.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh"
#include "prims/update_edge_src_dst_property.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugraph {

namespace {

template <typename vertex_t>
struct invalid_or_outside_local_vertex_partition_range_t {
  vertex_t num_vertices{};
  vertex_t local_vertex_partition_range_first{};
  vertex_t local_vertex_partition_range_last{};

  __device__ bool operator()(vertex_t v) const
  {
    return !is_valid_vertex(num_vertices, v) || (v < local_vertex_partition_range_first) ||
           (v >= local_vertex_partition_range_last);
  }
};

template <typename edge_t>
struct is_two_or_greater_t {
  __device__ bool operator()(edge_t core_number) const { return core_number >= edge_t{2}; }
};

template <typename vertex_t, typename edge_t>
struct extract_low_to_high_degree_edges_e_op_t {
  __device__ thrust::tuple<vertex_t, vertex_t> operator()(vertex_t src,
                                                          vertex_t dst,
                                                          edge_t src_out_degree,
                                                          edge_t dst_out_degree,
                                                          cuda::std::nullopt_t) const
  {
    return thrust::make_tuple(src, dst);
  }
};

template <typename vertex_t, typename edge_t>
struct extract_low_to_high_degree_edges_pred_op_t {
  __device__ bool operator()(vertex_t src,
                             vertex_t dst,
                             edge_t src_out_degree,
                             edge_t dst_out_degree,
                             cuda::std::nullopt_t) const
  {
    return (src_out_degree < dst_out_degree) ? true
                                             : (((src_out_degree == dst_out_degree) &&
                                                 (src < dst) /* tie-breaking using vertex ID */)
                                                  ? true
                                                  : false);
  }
};

template <typename vertex_t, typename edge_t>
struct intersection_op_t {
  __device__ thrust::tuple<edge_t, edge_t, edge_t> operator()(
    vertex_t,
    vertex_t,
    cuda::std::nullopt_t,
    cuda::std::nullopt_t,
    raft::device_span<vertex_t const> intersection) const
  {
    return thrust::make_tuple(static_cast<edge_t>(intersection.size()),
                              static_cast<edge_t>(intersection.size()),
                              edge_t{1});
  }
};

template <typename vertex_t, typename edge_t>
struct vertex_to_count_t {
  raft::device_span<vertex_t const> sorted_local_vertices{};
  raft::device_span<edge_t const> local_counts{};

  __device__ edge_t operator()(vertex_t v) const
  {
    auto it = thrust::lower_bound(
      thrust::seq, sorted_local_vertices.begin(), sorted_local_vertices.end(), v);
    if ((it != sorted_local_vertices.end()) && (*it == v)) {
      return *(local_counts.begin() + cuda::std::distance(sorted_local_vertices.begin(), it));
    } else {
      return edge_t{0};
    }
  }
};

// FIXME: better move this elsewhere for reuse
template <typename vertex_t>
struct vertex_offset_from_vertex_t {
  vertex_t local_vertex_partition_range_first{};

  __device__ vertex_t operator()(vertex_t v) const
  {
    return v - local_vertex_partition_range_first;
  }
};

}  // namespace

template <typename vertex_t, typename edge_t, bool multi_gpu>
void triangle_count(raft::handle_t const& handle,
                    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                    std::optional<raft::device_span<vertex_t const>> vertices,
                    raft::device_span<edge_t> counts,
                    bool do_expensive_check)
{
  using weight_t = float;  // dummy

  // 1. Check input arguments.

  CUGRAPH_EXPECTS(
    graph_view.is_symmetric(),
    "Invalid input arguments: triangle_count currently supports undirected graphs only.");
  CUGRAPH_EXPECTS(
    !graph_view.is_multigraph(),
    "Invalid input arguments: triangle_count currently does not support multi-graphs.");
  if (vertices) {
    CUGRAPH_EXPECTS(counts.size() == (*vertices).size(),
                    "Invalid arguments: counts.size() does not coincide with (*vertices).size().");
  } else {
    CUGRAPH_EXPECTS(
      counts.size() == static_cast<size_t>(graph_view.local_vertex_partition_range_size()),
      "Invalid arguments: counts.size() does not coincide with the number of local vertices.");
  }

  if (do_expensive_check) {
    if (vertices) {
      auto num_invalids =
        thrust::count_if(handle.get_thrust_policy(),
                         (*vertices).begin(),
                         (*vertices).end(),
                         invalid_or_outside_local_vertex_partition_range_t<vertex_t>{
                           graph_view.number_of_vertices(),
                           graph_view.local_vertex_partition_range_first(),
                           graph_view.local_vertex_partition_range_last()});

      if constexpr (multi_gpu) {
        auto& comm = handle.get_comms();
        num_invalids =
          host_scalar_allreduce(comm, num_invalids, raft::comms::op_t::SUM, handle.get_stream());
      }
      CUGRAPH_EXPECTS(num_invalids == 0,
                      "Invalid input arguments: invalid vertex IDs in *vertices.");
    }
  }

  if (vertices.has_value()) {
    auto aggregate_vertex_count =
      multi_gpu
        ? host_scalar_allreduce(
            handle.get_comms(), (*vertices).size(), raft::comms::op_t::SUM, handle.get_stream())
        : (*vertices).size();
    if (aggregate_vertex_count == 0) { return; }
  }

  auto cur_graph_view = graph_view;

  auto unmasked_cur_graph_view = cur_graph_view;
  if (unmasked_cur_graph_view.has_edge_mask()) { unmasked_cur_graph_view.clear_edge_mask(); }

  // 2. Mask out the edges that has source or destination that cannot be reached from vertices
  // within two hop (if vertices.has_value() is true).

  cugraph::edge_property_t<decltype(cur_graph_view), bool> edge_mask(handle);

  if (vertices) {
    cugraph::edge_property_t<decltype(cur_graph_view), bool> within_two_hop_edge_mask(
      handle, cur_graph_view);
    cugraph::fill_edge_property(
      handle, unmasked_cur_graph_view, within_two_hop_edge_mask.mutable_view(), false);

    rmm::device_uvector<vertex_t> unique_vertices((*vertices).size(), handle.get_stream());
    thrust::copy(
      handle.get_thrust_policy(), (*vertices).begin(), (*vertices).end(), unique_vertices.begin());
    thrust::sort(handle.get_thrust_policy(), unique_vertices.begin(), unique_vertices.end());
    unique_vertices.resize(
      cuda::std::distance(
        unique_vertices.begin(),
        thrust::unique(handle.get_thrust_policy(), unique_vertices.begin(), unique_vertices.end())),
      handle.get_stream());

    rmm::device_uvector<vertex_t> one_hop_nbrs(0, handle.get_stream());
    std::tie(std::ignore, one_hop_nbrs) = cugraph::k_hop_nbrs(
      handle,
      cur_graph_view,
      raft::device_span<vertex_t const>(unique_vertices.data(), unique_vertices.size()),
      size_t{1});

    rmm::device_uvector<vertex_t> unique_one_hop_nbrs(one_hop_nbrs.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 one_hop_nbrs.begin(),
                 one_hop_nbrs.end(),
                 unique_one_hop_nbrs.begin());
    one_hop_nbrs.resize(0, handle.get_stream());
    one_hop_nbrs.shrink_to_fit(handle.get_stream());
    thrust::sort(
      handle.get_thrust_policy(), unique_one_hop_nbrs.begin(), unique_one_hop_nbrs.end());
    unique_one_hop_nbrs.resize(cuda::std::distance(unique_one_hop_nbrs.begin(),
                                                   thrust::unique(handle.get_thrust_policy(),
                                                                  unique_one_hop_nbrs.begin(),
                                                                  unique_one_hop_nbrs.end())),
                               handle.get_stream());

    if constexpr (multi_gpu) {
      unique_one_hop_nbrs = detail::shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
        handle, std::move(unique_one_hop_nbrs), cur_graph_view.vertex_partition_range_lasts());
      thrust::sort(
        handle.get_thrust_policy(), unique_one_hop_nbrs.begin(), unique_one_hop_nbrs.end());
      unique_one_hop_nbrs.resize(cuda::std::distance(unique_one_hop_nbrs.begin(),
                                                     thrust::unique(handle.get_thrust_policy(),
                                                                    unique_one_hop_nbrs.begin(),
                                                                    unique_one_hop_nbrs.end())),
                                 handle.get_stream());
    }

    rmm::device_uvector<vertex_t> two_hop_nbrs(0, handle.get_stream());
    std::tie(std::ignore, two_hop_nbrs) = cugraph::k_hop_nbrs(
      handle,
      cur_graph_view,
      raft::device_span<vertex_t const>(unique_one_hop_nbrs.data(), unique_one_hop_nbrs.size()),
      size_t{1});

    rmm::device_uvector<vertex_t> unique_two_hop_nbrs(two_hop_nbrs.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 two_hop_nbrs.begin(),
                 two_hop_nbrs.end(),
                 unique_two_hop_nbrs.begin());
    two_hop_nbrs.resize(0, handle.get_stream());
    two_hop_nbrs.shrink_to_fit(handle.get_stream());
    thrust::sort(
      handle.get_thrust_policy(), unique_two_hop_nbrs.begin(), unique_two_hop_nbrs.end());
    unique_two_hop_nbrs.resize(cuda::std::distance(unique_two_hop_nbrs.begin(),
                                                   thrust::unique(handle.get_thrust_policy(),
                                                                  unique_two_hop_nbrs.begin(),
                                                                  unique_two_hop_nbrs.end())),
                               handle.get_stream());

    if constexpr (multi_gpu) {
      unique_two_hop_nbrs = detail::shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
        handle, std::move(unique_two_hop_nbrs), cur_graph_view.vertex_partition_range_lasts());
      thrust::sort(
        handle.get_thrust_policy(), unique_two_hop_nbrs.begin(), unique_two_hop_nbrs.end());
      unique_two_hop_nbrs.resize(cuda::std::distance(unique_two_hop_nbrs.begin(),
                                                     thrust::unique(handle.get_thrust_policy(),
                                                                    unique_two_hop_nbrs.begin(),
                                                                    unique_two_hop_nbrs.end())),
                                 handle.get_stream());
    }

    rmm::device_uvector<bool> within_two_hop_flags(
      cur_graph_view.local_vertex_partition_range_size(), handle.get_stream());

    thrust::fill(
      handle.get_thrust_policy(), within_two_hop_flags.begin(), within_two_hop_flags.end(), false);
    thrust::for_each(handle.get_thrust_policy(),
                     unique_vertices.begin(),
                     unique_vertices.end(),
                     [within_two_hop_flags = raft::device_span<bool>(within_two_hop_flags.data(),
                                                                     within_two_hop_flags.size()),
                      local_vertex_partition_range_first =
                        cur_graph_view.local_vertex_partition_range_first()] __device__(auto v) {
                       auto v_offset                  = v - local_vertex_partition_range_first;
                       within_two_hop_flags[v_offset] = true;
                     });
    unique_vertices.resize(0, handle.get_stream());
    unique_vertices.shrink_to_fit(handle.get_stream());

    thrust::for_each(handle.get_thrust_policy(),
                     unique_one_hop_nbrs.begin(),
                     unique_one_hop_nbrs.end(),
                     [within_two_hop_flags = raft::device_span<bool>(within_two_hop_flags.data(),
                                                                     within_two_hop_flags.size()),
                      local_vertex_partition_range_first =
                        cur_graph_view.local_vertex_partition_range_first()] __device__(auto v) {
                       auto v_offset                  = v - local_vertex_partition_range_first;
                       within_two_hop_flags[v_offset] = true;
                     });
    unique_one_hop_nbrs.resize(0, handle.get_stream());
    unique_one_hop_nbrs.shrink_to_fit(handle.get_stream());

    thrust::for_each(handle.get_thrust_policy(),
                     unique_two_hop_nbrs.begin(),
                     unique_two_hop_nbrs.end(),
                     [within_two_hop_flags = raft::device_span<bool>(within_two_hop_flags.data(),
                                                                     within_two_hop_flags.size()),
                      local_vertex_partition_range_first =
                        cur_graph_view.local_vertex_partition_range_first()] __device__(auto v) {
                       auto v_offset                  = v - local_vertex_partition_range_first;
                       within_two_hop_flags[v_offset] = true;
                     });
    unique_two_hop_nbrs.resize(0, handle.get_stream());
    unique_two_hop_nbrs.shrink_to_fit(handle.get_stream());

    edge_src_property_t<decltype(cur_graph_view), bool> edge_src_within_two_hop_flags(
      handle, cur_graph_view);
    edge_dst_property_t<decltype(cur_graph_view), bool> edge_dst_within_two_hop_flags(
      handle, cur_graph_view);
    update_edge_src_property(handle,
                             cur_graph_view,
                             within_two_hop_flags.begin(),
                             edge_src_within_two_hop_flags.mutable_view());
    update_edge_dst_property(handle,
                             cur_graph_view,
                             within_two_hop_flags.begin(),
                             edge_dst_within_two_hop_flags.mutable_view());

    transform_e(
      handle,
      cur_graph_view,
      edge_src_within_two_hop_flags.view(),
      edge_dst_within_two_hop_flags.view(),
      edge_dummy_property_t{}.view(),
      [] __device__(auto, auto, auto src_within_two_hop, auto dst_within_two_hop, auto) {
        return src_within_two_hop && dst_within_two_hop;
      },
      within_two_hop_edge_mask.mutable_view());

    edge_mask = std::move(within_two_hop_edge_mask);
    if (cur_graph_view.has_edge_mask()) { cur_graph_view.clear_edge_mask(); }
    cur_graph_view.attach_edge_mask(edge_mask.view());
  }

  // 3. Exclude self-loops

  {
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

    edge_mask = std::move(self_loop_edge_mask);
    if (cur_graph_view.has_edge_mask()) { cur_graph_view.clear_edge_mask(); }
    cur_graph_view.attach_edge_mask(edge_mask.view());
  }

  // 4. Find 2-core and exclude edges that do not belong to 2-core add masking support).

  {
    cugraph::edge_property_t<decltype(cur_graph_view), bool> in_two_core_edge_mask(handle,
                                                                                   cur_graph_view);
    cugraph::fill_edge_property(
      handle, unmasked_cur_graph_view, in_two_core_edge_mask.mutable_view(), false);

    rmm::device_uvector<edge_t> core_numbers(cur_graph_view.number_of_vertices(),
                                             handle.get_stream());
    core_number(
      handle, cur_graph_view, core_numbers.data(), k_core_degree_type_t::OUT, size_t{2}, size_t{2});

    edge_src_property_t<decltype(cur_graph_view), bool> edge_src_in_two_cores(handle,
                                                                              cur_graph_view);
    edge_dst_property_t<decltype(cur_graph_view), bool> edge_dst_in_two_cores(handle,
                                                                              cur_graph_view);
    auto in_two_core_first =
      thrust::make_transform_iterator(core_numbers.begin(), is_two_or_greater_t<edge_t>{});
    rmm::device_uvector<bool> in_two_core_flags(core_numbers.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 in_two_core_first,
                 in_two_core_first + core_numbers.size(),
                 in_two_core_flags.begin());
    update_edge_src_property(
      handle, cur_graph_view, in_two_core_flags.begin(), edge_src_in_two_cores.mutable_view());
    update_edge_dst_property(
      handle, cur_graph_view, in_two_core_flags.begin(), edge_dst_in_two_cores.mutable_view());

    transform_e(
      handle,
      cur_graph_view,
      edge_src_in_two_cores.view(),
      edge_dst_in_two_cores.view(),
      edge_dummy_property_t{}.view(),
      [] __device__(auto, auto, auto src_in_two_core, auto dst_in_two_core, auto) {
        return src_in_two_core && dst_in_two_core;
      },
      in_two_core_edge_mask.mutable_view());

    edge_mask = std::move(in_two_core_edge_mask);
    if (cur_graph_view.has_edge_mask()) { cur_graph_view.clear_edge_mask(); }
    cur_graph_view.attach_edge_mask(edge_mask.view());
  }

  // 5. Keep only the edges from a low-degree vertex to a high-degree vertex.

  graph_t<vertex_t, edge_t, false, multi_gpu> modified_graph(handle);
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

  {
    auto out_degrees = cur_graph_view.compute_out_degrees(handle);

    edge_src_property_t<decltype(cur_graph_view), edge_t> edge_src_out_degrees(handle,
                                                                               cur_graph_view);
    edge_dst_property_t<decltype(cur_graph_view), edge_t> edge_dst_out_degrees(handle,
                                                                               cur_graph_view);
    update_edge_src_property(
      handle, cur_graph_view, out_degrees.begin(), edge_src_out_degrees.mutable_view());
    update_edge_dst_property(
      handle, cur_graph_view, out_degrees.begin(), edge_dst_out_degrees.mutable_view());
    auto [srcs, dsts] =
      extract_transform_if_e(handle,
                             cur_graph_view,
                             edge_src_out_degrees.view(),
                             edge_dst_out_degrees.view(),
                             edge_dummy_property_t{}.view(),
                             extract_low_to_high_degree_edges_e_op_t<vertex_t, edge_t>{},
                             extract_low_to_high_degree_edges_pred_op_t<vertex_t, edge_t>{});

    if constexpr (multi_gpu) {
      std::tie(
        srcs, dsts, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) =
        shuffle_ext_edges<vertex_t, edge_t, weight_t, int32_t, int32_t>(handle,
                                                                        std::move(srcs),
                                                                        std::move(dsts),
                                                                        std::nullopt,
                                                                        std::nullopt,
                                                                        std::nullopt,
                                                                        std::nullopt,
                                                                        std::nullopt,
                                                                        false);
    }

    std::tie(modified_graph, std::ignore, std::ignore, std::ignore, renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{false /* now asymmetric */, cur_graph_view.is_multigraph()},
        true);
  }

  cur_graph_view = modified_graph.view();

  // 6. neighbor intersection

  rmm::device_uvector<edge_t> cur_graph_counts(size_t{0}, handle.get_stream());
  {
    cur_graph_counts.resize(cur_graph_view.local_vertex_partition_range_size(),
                            handle.get_stream());

    transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v(handle,
                                                              cur_graph_view,
                                                              edge_src_dummy_property_t{}.view(),
                                                              edge_dst_dummy_property_t{}.view(),
                                                              intersection_op_t<vertex_t, edge_t>{},
                                                              edge_t{0},
                                                              cur_graph_counts.begin(),
                                                              do_expensive_check);
  }

  // 7. update counts

  {
    thrust::fill(handle.get_thrust_policy(), counts.begin(), counts.end(), edge_t{0});
    auto local_vertices = std::move(*renumber_map);
    auto local_counts   = std::move(cur_graph_counts);

    if constexpr (multi_gpu) {
      auto& comm       = handle.get_comms();
      auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      auto h_vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
      rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
        h_vertex_partition_range_lasts.size(), handle.get_stream());
      raft::update_device(d_vertex_partition_range_lasts.data(),
                          h_vertex_partition_range_lasts.data(),
                          h_vertex_partition_range_lasts.size(),
                          handle.get_stream());

      rmm::device_uvector<vertex_t> rx_local_vertices(size_t{0}, handle.get_stream());
      rmm::device_uvector<edge_t> rx_local_counts(size_t{0}, handle.get_stream());
      std::tie(rx_local_vertices, rx_local_counts, std::ignore) =
        groupby_gpu_id_and_shuffle_kv_pairs(
          handle.get_comms(),
          local_vertices.begin(),
          local_vertices.end(),
          local_counts.begin(),
          cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
            raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                              d_vertex_partition_range_lasts.size()),
            major_comm_size,
            minor_comm_size},
          handle.get_stream());

      local_vertices = std::move(rx_local_vertices);
      local_counts   = std::move(rx_local_counts);
    }
    thrust::sort_by_key(handle.get_thrust_policy(),
                        local_vertices.begin(),
                        local_vertices.end(),
                        local_counts.begin());

    if (vertices) {
      thrust::transform(
        handle.get_thrust_policy(),
        (*vertices).begin(),
        (*vertices).end(),
        counts.begin(),
        vertex_to_count_t<vertex_t, edge_t>{
          raft::device_span<vertex_t const>(local_vertices.begin(), local_vertices.end()),
          raft::device_span<edge_t const>(local_counts.begin(), local_counts.end())});
    } else {
      thrust::scatter(
        handle.get_thrust_policy(),
        local_counts.begin(),
        local_counts.end(),
        thrust::make_transform_iterator(
          local_vertices.begin(),
          vertex_offset_from_vertex_t<vertex_t>{graph_view.local_vertex_partition_range_first()}),
        counts.begin());
    }
  }

  return;
}

}  // namespace cugraph
