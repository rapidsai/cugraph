/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <prims/extract_if_e.cuh>
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

template <typename vertex_t>
struct is_not_self_loop_t {
  __device__ bool operator()(
    vertex_t src, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
  {
    return src != dst;
  }
};

template <typename edge_t>
struct is_two_or_greater_t {
  __device__ uint8_t operator()(edge_t core_number) const
  {
    return core_number >= edge_t{2} ? uint8_t{1} : uint8_t{0};
  }
};

template <typename vertex_t>
struct in_two_core_t {
  __device__ bool operator()(
    vertex_t, vertex_t, uint8_t src_in_two_core, uint8_t dst_in_two_core, thrust::nullopt_t) const
  {
    return (src_in_two_core == uint8_t{1}) && (dst_in_two_core == uint8_t{1});
  }
};

template <typename vertex_t, typename edge_t>
struct low_to_high_degree_t {
  __device__ bool operator()(vertex_t src,
                             vertex_t dst,
                             edge_t src_out_degree,
                             edge_t dst_out_degree,
                             thrust::nullopt_t) const
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
    thrust::nullopt_t,
    thrust::nullopt_t,
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
      return *(local_counts.begin() + thrust::distance(sorted_local_vertices.begin(), it));
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

  // FIXME: if vertices.has_value(), we may better work with the subgraph including only the
  // neighbors within two-hop.

  // 2. Exclude self-loops (FIXME: better mask-out once we add masking support).

  std::optional<graph_t<vertex_t, edge_t, false, multi_gpu>> modified_graph{std::nullopt};
  std::optional<graph_view_t<vertex_t, edge_t, false, multi_gpu>> modified_graph_view{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

  if (graph_view.count_self_loops(handle) > edge_t{0}) {
    auto [srcs, dsts] = extract_if_e(handle,
                                     graph_view,
                                     edge_src_dummy_property_t{}.view(),
                                     edge_dst_dummy_property_t{}.view(),
                                     is_not_self_loop_t<vertex_t>{});

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                           edge_t,
                                                                           weight_t,
                                                                           int32_t>(
          handle, std::move(srcs), std::move(dsts), std::nullopt, std::nullopt);
    }

    std::tie(*modified_graph, std::ignore, std::ignore, renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{true, graph_view.is_multigraph()},
        true);

    modified_graph_view = (*modified_graph).view();
  }

  // 3. Find 2-core and exclude edges that do not belong to 2-core (FIXME: better mask-out once we
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
      handle, cur_graph_view, core_numbers.data(), k_core_degree_type_t::OUT, size_t{2}, size_t{2});

    edge_src_property_t<decltype(cur_graph_view), uint8_t> edge_src_in_two_cores(handle,
                                                                                 cur_graph_view);
    edge_dst_property_t<decltype(cur_graph_view), uint8_t> edge_dst_in_two_cores(handle,
                                                                                 cur_graph_view);
    auto in_two_core_first =
      thrust::make_transform_iterator(core_numbers.begin(), is_two_or_greater_t<edge_t>{});
    rmm::device_uvector<uint8_t> in_two_core_flags(core_numbers.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 in_two_core_first,
                 in_two_core_first + core_numbers.size(),
                 in_two_core_flags.begin());
    update_edge_src_property(
      handle, cur_graph_view, in_two_core_flags.begin(), edge_src_in_two_cores);
    update_edge_dst_property(
      handle, cur_graph_view, in_two_core_flags.begin(), edge_dst_in_two_cores);
    auto [srcs, dsts] = extract_if_e(handle,
                                     cur_graph_view,
                                     edge_src_in_two_cores.view(),
                                     edge_dst_in_two_cores.view(),
                                     in_two_core_t<vertex_t>{});

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                           edge_t,
                                                                           weight_t,
                                                                           int32_t>(
          handle, std::move(srcs), std::move(dsts), std::nullopt, std::nullopt);
    }

    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};
    std::tie(*modified_graph, std::ignore, std::ignore, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
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
    auto [srcs, dsts] = extract_if_e(handle,
                                     cur_graph_view,
                                     edge_src_out_degrees.view(),
                                     edge_dst_out_degrees.view(),
                                     low_to_high_degree_t<vertex_t, edge_t>{});

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                           edge_t,
                                                                           weight_t,
                                                                           int32_t>(
          handle, std::move(srcs), std::move(dsts), std::nullopt, std::nullopt);
    }

    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};
    std::tie(*modified_graph, std::ignore, std::ignore, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
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

  // 5. neighbor intersection

  rmm::device_uvector<edge_t> cur_graph_counts(size_t{0}, handle.get_stream());
  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;
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

  // 6. update counts

  {
    thrust::fill(handle.get_thrust_policy(), counts.begin(), counts.end(), edge_t{0});
    auto local_vertices = std::move(*renumber_map);
    auto local_counts   = std::move(cur_graph_counts);

    if constexpr (multi_gpu) {
      // FIXME: better refactor this shuffle for reuse
      auto& comm = handle.get_comms();

      thrust::sort_by_key(handle.get_thrust_policy(),
                          local_vertices.begin(),
                          local_vertices.end(),
                          local_counts.begin());
      auto h_vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
      rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
        h_vertex_partition_range_lasts.size(), handle.get_stream());
      raft::update_device(d_vertex_partition_range_lasts.data(),
                          h_vertex_partition_range_lasts.data(),
                          h_vertex_partition_range_lasts.size(),
                          handle.get_stream());
      rmm::device_uvector<size_t> d_lasts(d_vertex_partition_range_lasts.size(),
                                          handle.get_stream());
      thrust::lower_bound(handle.get_thrust_policy(),
                          local_vertices.begin(),
                          local_vertices.end(),
                          d_vertex_partition_range_lasts.begin(),
                          d_vertex_partition_range_lasts.end(),
                          d_lasts.begin());
      std::vector<size_t> h_lasts(d_lasts.size());
      raft::update_host(h_lasts.data(), d_lasts.data(), d_lasts.size(), handle.get_stream());
      handle.sync_stream();

      std::vector<size_t> tx_counts(h_lasts.size());
      std::adjacent_difference(h_lasts.begin(), h_lasts.end(), tx_counts.begin());

      rmm::device_uvector<vertex_t> rx_local_vertices(size_t{0}, handle.get_stream());
      rmm::device_uvector<edge_t> rx_local_counts(size_t{0}, handle.get_stream());
      std::tie(rx_local_vertices, std::ignore) =
        shuffle_values(comm, local_vertices.begin(), tx_counts, handle.get_stream());
      std::tie(rx_local_counts, std::ignore) =
        shuffle_values(comm, local_counts.begin(), tx_counts, handle.get_stream());

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
