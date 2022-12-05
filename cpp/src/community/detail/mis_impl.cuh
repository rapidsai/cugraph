
/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <community/detail/mis.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <prims/fill_edge_src_dst_property.cuh>

#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <raft/util/integer_utils.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/count.h>
#include <thrust/distance.h>

#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <numeric>
#include <type_traits>
#include <utility>

namespace cugraph {

namespace detail {

template <typename vertex_t>
rmm::device_uvector<vertex_t> select_a_random_set_of_vetices(raft::handle_t const& handle,
                                                             vertex_t begin,
                                                             vertex_t end,
                                                             vertex_t count,
                                                             uint64_t seed,
                                                             int repetitions_per_vertex = 0)
{
#if 0
  auto& comm                  = handle.get_comms();
  auto const comm_rank        = comm.get_rank();
#endif
  vertex_t number_of_vertices = end - begin;

  rmm::device_uvector<vertex_t> vertices(
    std::max((repetitions_per_vertex + 1) * number_of_vertices, count), handle.get_stream());
  thrust::tabulate(
    handle.get_thrust_policy(),
    vertices.begin(),
    vertices.end(),
    [begin, number_of_vertices] __device__(auto v) { return begin + (v % number_of_vertices); });
  thrust::default_random_engine g;
  g.seed(seed);
  thrust::shuffle(handle.get_thrust_policy(), vertices.begin(), vertices.end(), g);
  vertices.resize(count, handle.get_stream());
  return vertices;
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<vertex_t> compute_mis(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  using GraphViewType = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

  size_t number_of_vertices = graph_view.local_vertex_partition_range_size();

  rmm::device_uvector<vertex_t> mis(number_of_vertices, handle.get_stream());

  //
  // Flag to indicate if a vertex is included in MIS
  //
  rmm::device_uvector<uint8_t> mis_inclusion_flags(number_of_vertices, handle.get_stream());
  thrust::uninitialized_fill(
    handle.get_thrust_policy(), mis_inclusion_flags.begin(), mis_inclusion_flags.end(), uint8_t{0});

  //
  // Falg to indicate if a vertex should be discarded from consideration, either
  // because it's already in MIS or or one of its neighbor is in the MIS
  //
  rmm::device_uvector<uint8_t> discard_flags(number_of_vertices, handle.get_stream());
  thrust::uninitialized_fill(
    handle.get_thrust_policy(), discard_flags.begin(), discard_flags.end(), uint8_t{0});

  auto random_vs = call_a_function_that_does_not_exists(handle,
                       graph_view.local_vertex_partition_range_first(),
                       graph_view.local_vertex_partition_range_last(),
                       static_cast<vertex_t>(0.2 * graph_view.local_vertex_partition_range_size()),
                       0);

  auto random_vs2 = select_a_random_set_of_vetices_10(
    handle,
    graph_view.local_vertex_partition_range_first(),
    graph_view.local_vertex_partition_range_last(),
    static_cast<vertex_t>(0.2 * graph_view.local_vertex_partition_range_size()),
    0);

  while (true) {
    //
    // Select a random set of vertices
    //

    auto random_vertices = select_a_random_set_of_vetices(
      handle,
      graph_view.local_vertex_partition_range_first(),
      graph_view.local_vertex_partition_range_last(),
      static_cast<vertex_t>(0.2 * graph_view.local_vertex_partition_range_size()),
      0);

    thrust::sort(handle.get_thrust_policy(),
                 random_vertices.begin(),
                 random_vertices.end(),
                 thrust::less<vertex_t>());
    random_vertices.resize(
      static_cast<vertex_t>(thrust::distance(thrust::unique(handle.get_thrust_policy(),
                                                            random_vertices.begin(),
                                                            random_vertices.end(),
                                                            thrust::less<vertex_t>()),
                                             random_vertices.begin())),
      handle.get_stream());

    rmm::device_uvector<uint8_t> selection_flags(graph_view.local_vertex_partition_range_size(),
                                                 handle.get_stream());

    thrust::transform(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
      selection_flags.begin(),
      [selected_nodes    = random_vertices.data(),
       nr_selected_nodes = random_vertices.size()] __device__(auto node) {
        return thrust::binary_search(
          thrust::device, selected_nodes, selected_nodes + nr_selected_nodes, node);
      });

    edge_src_property_t<GraphViewType, uint8_t> src_inclusion_flag_cache(handle);
    edge_dst_property_t<GraphViewType, uint8_t> dst_inclusion_flag_cache(handle);

    edge_src_property_t<GraphViewType, uint8_t> src_selection_flag_cache(handle);
    edge_dst_property_t<GraphViewType, uint8_t> dst_selection_flag_cache(handle);

    // As there is only one outedge, compute_out_weight_sums would return weight of
    // outgoing edge.

    auto ranks = compute_out_weight_sums(handle, graph_view, *edge_weight_view);

    edge_src_property_t<GraphViewType, weight_t> src_rank_cache(handle);
    edge_dst_property_t<GraphViewType, weight_t> dst_rank_cache(handle);

    if constexpr (GraphViewType::is_multi_gpu) {
      src_inclusion_flag_cache = edge_src_property_t<GraphViewType, uint8_t>(handle, graph_view);
      dst_inclusion_flag_cache = edge_dst_property_t<GraphViewType, uint8_t>(handle, graph_view);

      update_edge_src_property(
        handle, graph_view, mis_inclusion_flags.begin(), src_inclusion_flag_cache);

      update_edge_dst_property(
        handle, graph_view, mis_inclusion_flags.begin(), dst_inclusion_flag_cache);

      src_selection_flag_cache = edge_src_property_t<GraphViewType, uint8_t>(handle, graph_view);
      dst_selection_flag_cache = edge_dst_property_t<GraphViewType, uint8_t>(handle, graph_view);
      update_edge_src_property(
        handle, graph_view, selection_flags.begin(), src_selection_flag_cache);
      update_edge_dst_property(
        handle, graph_view, selection_flags.begin(), dst_selection_flag_cache);

      src_rank_cache = edge_src_property_t<GraphViewType, weight_t>(handle, graph_view);
      dst_rank_cache = edge_dst_property_t<GraphViewType, weight_t>(handle, graph_view);
      update_edge_src_property(handle, graph_view, ranks.begin(), src_rank_cache);
      update_edge_dst_property(handle, graph_view, ranks.begin(), dst_rank_cache);
    }

    rmm::device_uvector<uint8_t> updated_selection_flags(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());

    per_v_transform_reduce_outgoing_e(
      handle,
      graph_view,
      GraphViewType::is_multi_gpu
        ? view_concat(
            src_inclusion_flag_cache.view(), src_selection_flag_cache.view(), src_rank_cache.view())
        : view_concat(
            detail::edge_major_property_view_t<vertex_t, uint8_t const*>(
              mis_inclusion_flags.data()),
            detail::edge_major_property_view_t<vertex_t, uint8_t const*>(selection_flags.data()),
            detail::edge_major_property_view_t<vertex_t, weight_t const*>(ranks.data())),
      GraphViewType::is_multi_gpu
        ? view_concat(
            dst_inclusion_flag_cache.view(), dst_selection_flag_cache.view(), dst_rank_cache.view())
        : view_concat(detail::edge_minor_property_view_t<vertex_t, uint8_t const*>(
                        mis_inclusion_flags.data(), vertex_t{0}),
                      detail::edge_minor_property_view_t<vertex_t, uint8_t const*>(
                        selection_flags.data(), vertex_t{0}),
                      detail::edge_minor_property_view_t<vertex_t, weight_t const*>(ranks.data(),
                                                                                    vertex_t{0})),
      [] __device__(auto src, auto dst, auto wt, auto src_info, auto dst_info) {
        auto is_src_selected = thrust::get<1>(src_info);
        auto src_rank        = thrust::get<2>(src_info);

        auto is_dst_included = thrust::get<0>(dst_info);
        auto is_dst_selected = thrust::get<1>(dst_info);
        auto dst_rank        = thrust::get<2>(dst_info);

        // If a neighbor is already included into MIS,
        // then it can't be included, discard if forever.

        if (is_dst_included) return thrust::make_tuple(uint8_t{0}, uint8_t{1});

        //
        // Give priority to high rank, high id vertex
        //

        // if (is_src_selected && is_dst_selected)
        //   return thrust::make_tuple(src_rank, src) > thrust::make_tuple(dst_rank, dst);

        if (is_src_selected && is_dst_selected) {
          if (src_rank < dst_rank) {
            return thrust::make_tuple(uint8_t{0}, uint8_t{0});  // unselect v
          } else if (fabs(src_rank - dst_rank) < 1e-8) {
            if (src < dst) {
              return thrust::make_tuple(uint8_t{0}, uint8_t{0});  // unselect v
            }
          }
        }

        return thrust::make_tuple(is_src_selected, uint8_t{0});
      },
      thrust::make_tuple(uint8_t{0}, uint8_t{0}),
      thrust::make_zip_iterator(
        thrust::make_tuple(updated_selection_flags.begin(), discard_flags.begin())));

    if constexpr (GraphViewType::is_multi_gpu) {
      src_selection_flag_cache = edge_src_property_t<GraphViewType, uint8_t>(handle, graph_view);
      dst_selection_flag_cache = edge_dst_property_t<GraphViewType, uint8_t>(handle, graph_view);
      update_edge_src_property(
        handle, graph_view, updated_selection_flags.begin(), src_selection_flag_cache);
      update_edge_dst_property(
        handle, graph_view, updated_selection_flags.begin(), dst_selection_flag_cache);
    }

    rmm::device_uvector<uint8_t> final_selection_flags(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());

    per_v_transform_reduce_incoming_e(
      handle,
      graph_view,
      GraphViewType::is_multi_gpu
        ? view_concat(
            src_inclusion_flag_cache.view(), src_selection_flag_cache.view(), src_rank_cache.view())
        : view_concat(detail::edge_major_property_view_t<vertex_t, uint8_t const*>(
                        mis_inclusion_flags.data()),
                      detail::edge_major_property_view_t<vertex_t, uint8_t const*>(
                        updated_selection_flags.data()),
                      detail::edge_major_property_view_t<vertex_t, weight_t const*>(ranks.data())),
      GraphViewType::is_multi_gpu
        ? view_concat(
            dst_inclusion_flag_cache.view(), dst_selection_flag_cache.view(), dst_rank_cache.view())
        : view_concat(

            detail::edge_minor_property_view_t<vertex_t, uint8_t const*>(mis_inclusion_flags.data(),
                                                                         vertex_t{0}),
            detail::edge_minor_property_view_t<vertex_t, uint8_t const*>(
              updated_selection_flags.data(), vertex_t{0}),
            detail::edge_minor_property_view_t<vertex_t, weight_t const*>(ranks.data(),
                                                                          vertex_t{0})),
      [] __device__(auto src, auto dst, auto wt, auto src_info, auto dst_info) {
        auto is_src_included = thrust::get<0>(src_info);
        auto is_src_selected = thrust::get<1>(src_info);
        auto src_rank        = thrust::get<2>(src_info);

        auto is_dst_selected = thrust::get<1>(dst_info);
        auto dst_rank        = thrust::get<2>(dst_info);

        if (is_src_included) thrust::make_tuple(uint8_t{0}, uint8_t{0});

        // Give priority to high rank, high id vertex
        if (is_src_selected && is_dst_selected)
          return thrust::make_tuple(static_cast<uint8_t>(thrust::make_tuple(dst_rank, dst) >
                                                         thrust::make_tuple(src_rank, src)),
                                    uint8_t{0});

        return thrust::make_tuple(is_dst_selected, uint8_t{0});
      },
      thrust::make_tuple(uint8_t{0}, uint8_t{0}),
      thrust::make_zip_iterator(
        thrust::make_tuple(final_selection_flags.begin(), discard_flags.begin())));

    // Add selected vertices to MIS and discard in the next selection round
    thrust::transform(handle.get_thrust_policy(),
                      final_selection_flags.begin(),
                      final_selection_flags.end(),
                      thrust::make_zip_iterator(
                        thrust::make_tuple(mis_inclusion_flags.begin(), discard_flags.begin())),
                      thrust::make_zip_iterator(
                        thrust::make_tuple(mis_inclusion_flags.begin(), discard_flags.begin())),
                      [] __device__(auto selection_flag, auto prev_flag_pair) {
                        auto prev_inclusion_flag = thrust::get<0>(prev_flag_pair);
                        auto prev_discard_flag   = thrust::get<1>(prev_flag_pair);

                        return thrust::make_tuple(
                          static_cast<uint8_t>(selection_flag || prev_inclusion_flag),
                          static_cast<uint8_t>(selection_flag || prev_discard_flag));
                      });

    size_t nr_include_vertices   = thrust::count_if(handle.get_thrust_policy(),
                                                  mis_inclusion_flags.begin(),
                                                  mis_inclusion_flags.end(),
                                                  [] __device__(auto flag) { return flag > 0; });
    size_t nr_discarded_vertices = thrust::count_if(handle.get_thrust_policy(),
                                                    discard_flags.begin(),
                                                    discard_flags.end(),
                                                    [] __device__(auto flag) { return flag > 0; });

    if ((number_of_vertices - nr_include_vertices - nr_discarded_vertices) == 0) break;
  }

  size_t nr_vertices_in_mis = thrust::count_if(handle.get_thrust_policy(),
                                               mis_inclusion_flags.begin(),
                                               mis_inclusion_flags.end(),
                                               [] __device__(auto flag) { return flag > 0; });

  thrust::copy_if(handle.get_thrust_policy(),
                  thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
                  thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
                  mis_inclusion_flags.begin(),
                  mis.begin(),
                  [] __device__(auto flag) { return flag > 0; });

  mis.resize(nr_vertices_in_mis, handle.get_stream());

  return mis;
}
}  // namespace detail
}  // namespace cugraph