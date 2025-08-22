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

#include "detail/shuffle_wrappers.hpp"
#include "sampling/detail/sampling_utils.hpp"

#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <optional>
#include <span>
#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename label_t, typename edge_time_t>
std::tuple<rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<label_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<std::tuple<rmm::device_uvector<vertex_t>,
                                    std::optional<rmm::device_uvector<label_t>>,
                                    std::optional<rmm::device_uvector<edge_time_t>>>>>
prepare_next_frontier(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> sampled_src_vertices,
  std::optional<raft::device_span<label_t const>> sampled_src_vertex_labels,
  std::optional<raft::device_span<edge_time_t const>> sampled_src_vertex_times,
  raft::host_span<raft::device_span<vertex_t const>> sampled_dst_vertices,
  std::optional<raft::host_span<raft::device_span<label_t const>>> sampled_dst_vertex_labels,
  std::optional<raft::host_span<raft::device_span<edge_time_t const>>> sampled_dst_vertex_times,
  std::optional<std::tuple<rmm::device_uvector<vertex_t>,
                           std::optional<rmm::device_uvector<label_t>>,
                           std::optional<rmm::device_uvector<edge_time_t>>>>&&
    vertex_used_as_source,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool multi_gpu,
  bool do_expensive_check)
{
  size_t frontier_size = std::transform_reduce(sampled_dst_vertices.begin(),
                                               sampled_dst_vertices.end(),
                                               size_t{0},
                                               std::plus{},
                                               [](auto span) { return span.size(); });

  if (prior_sources_behavior == prior_sources_behavior_t::CARRY_OVER) {
    frontier_size += sampled_src_vertices.size();
  }

  rmm::device_uvector<vertex_t> frontier_vertices(frontier_size, handle.get_stream());
  auto frontier_vertex_labels =
    sampled_dst_vertex_labels
      ? std::make_optional<rmm::device_uvector<label_t>>(frontier_size, handle.get_stream())
      : std::nullopt;
  auto frontier_vertex_times =
    sampled_dst_vertex_times
      ? std::make_optional<rmm::device_uvector<edge_time_t>>(frontier_size, handle.get_stream())
      : std::nullopt;

  size_t current_pos = 0;

  if (prior_sources_behavior == prior_sources_behavior_t::CARRY_OVER) {
    thrust::copy(handle.get_thrust_policy(),
                 sampled_src_vertices.begin(),
                 sampled_src_vertices.end(),
                 frontier_vertices.begin());
    current_pos = sampled_src_vertices.size();
  }

  std::for_each(sampled_dst_vertices.begin(),
                sampled_dst_vertices.end(),
                [&handle, &frontier_vertices, &current_pos](auto& list) {
                  thrust::copy(handle.get_thrust_policy(),
                               list.begin(),
                               list.end(),
                               frontier_vertices.begin() + current_pos);
                  current_pos += list.size();
                });

  if (frontier_vertex_labels) {
    current_pos = 0;
    if (prior_sources_behavior == prior_sources_behavior_t::CARRY_OVER) {
      thrust::copy(handle.get_thrust_policy(),
                   sampled_src_vertex_labels->begin(),
                   sampled_src_vertex_labels->end(),
                   frontier_vertex_labels->begin());
      current_pos = sampled_src_vertex_labels->size();
    }

    std::for_each(sampled_dst_vertex_labels->begin(),
                  sampled_dst_vertex_labels->end(),
                  [&handle, &frontier_vertex_labels, &current_pos](auto& list) {
                    if (list.size() > 0)
                      thrust::copy(handle.get_thrust_policy(),
                                   list.begin(),
                                   list.end(),
                                   frontier_vertex_labels->begin() + current_pos);
                    current_pos += list.size();
                  });
  }

  if (frontier_vertex_times) {
    current_pos = 0;
    if (prior_sources_behavior == prior_sources_behavior_t::CARRY_OVER) {
      thrust::copy(handle.get_thrust_policy(),
                   sampled_src_vertex_times->begin(),
                   sampled_src_vertex_times->end(),
                   frontier_vertex_times->begin());
      current_pos = sampled_src_vertex_times->size();
    }

    std::for_each(sampled_dst_vertex_times->begin(),
                  sampled_dst_vertex_times->end(),
                  [&handle, &frontier_vertex_times, &current_pos](auto& list) {
                    if (list.size() > 0)
                      thrust::copy(handle.get_thrust_policy(),
                                   list.begin(),
                                   list.end(),
                                   frontier_vertex_times->begin() + current_pos);
                    current_pos += list.size();
                  });
  }

  if (multi_gpu) {
    if (frontier_vertex_labels) {
      if (frontier_vertex_times) {
        std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
        vertex_properties.push_back(std::move(*frontier_vertex_labels));
        vertex_properties.push_back(std::move(*frontier_vertex_times));
        std::tie(frontier_vertices, vertex_properties) =
          shuffle_int_vertices(handle,
                               std::move(frontier_vertices),
                               std::move(vertex_properties),
                               vertex_partition_range_lasts);
        frontier_vertex_labels =
          std::move(std::get<rmm::device_uvector<label_t>>(vertex_properties[0]));
        frontier_vertex_times =
          std::move(std::get<rmm::device_uvector<edge_time_t>>(vertex_properties[1]));
      } else {
        std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
        vertex_properties.push_back(std::move(*frontier_vertex_labels));
        std::tie(frontier_vertices, vertex_properties) =
          shuffle_int_vertices(handle,
                               std::move(frontier_vertices),
                               std::move(vertex_properties),
                               vertex_partition_range_lasts);
        frontier_vertex_labels =
          std::move(std::get<rmm::device_uvector<label_t>>(vertex_properties[0]));
      }
    } else {
      if (frontier_vertex_times) {
        std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
        vertex_properties.push_back(std::move(*frontier_vertex_times));
        std::tie(frontier_vertices, vertex_properties) =
          shuffle_int_vertices(handle,
                               std::move(frontier_vertices),
                               std::move(vertex_properties),
                               vertex_partition_range_lasts);
        frontier_vertex_times =
          std::move(std::get<rmm::device_uvector<edge_time_t>>(vertex_properties[0]));
      } else {
        std::tie(frontier_vertices, std::ignore) =
          shuffle_int_vertices(handle,
                               std::move(frontier_vertices),
                               std::vector<cugraph::arithmetic_device_uvector_t>{},
                               vertex_partition_range_lasts);
      }
    }
  }

  if (frontier_vertex_labels) {
    auto begin_iter =
      thrust::make_zip_iterator(frontier_vertices.begin(), frontier_vertex_labels->begin());
    if (frontier_vertex_times) {
      thrust::sort_by_key(handle.get_thrust_policy(),
                          begin_iter,
                          begin_iter + frontier_vertices.size(),
                          frontier_vertex_times->begin());

    } else {
      thrust::sort(handle.get_thrust_policy(), begin_iter, begin_iter + frontier_vertices.size());
    }
  } else {
    if (frontier_vertex_times) {
      thrust::sort_by_key(handle.get_thrust_policy(),
                          frontier_vertices.begin(),
                          frontier_vertices.end(),
                          frontier_vertex_times->begin());

    } else {
      thrust::sort(handle.get_thrust_policy(), frontier_vertices.begin(), frontier_vertices.end());
    }
  }

  if (vertex_used_as_source) {
    auto& [verts, labels, times] = *vertex_used_as_source;

    // add sources from this expansion to the vertex_used_as_source
    size_t current_verts_size = verts.size();
    size_t new_verts_size     = current_verts_size + sampled_src_vertices.size();

    verts.resize(new_verts_size, handle.get_stream());

    thrust::copy(handle.get_thrust_policy(),
                 sampled_src_vertices.begin(),
                 sampled_src_vertices.end(),
                 verts.begin() + current_verts_size);

    // sort and unique the vertex_used_as_source structures
    if (sampled_src_vertex_labels) {
      labels->resize(new_verts_size, handle.get_stream());

      thrust::copy(handle.get_thrust_policy(),
                   sampled_src_vertex_labels->begin(),
                   sampled_src_vertex_labels->end(),
                   labels->begin() + current_verts_size);
    }

    if (sampled_src_vertex_times) {
      times->resize(new_verts_size, handle.get_stream());

      thrust::copy(handle.get_thrust_policy(),
                   sampled_src_vertex_times->begin(),
                   sampled_src_vertex_times->end(),
                   times->begin() + current_verts_size);
    }

    if (sampled_src_vertex_labels) {
      if (sampled_src_vertex_times) {
        auto begin_iter = thrust::make_zip_iterator(verts.begin(), labels->begin(), times->begin());

        thrust::sort(handle.get_thrust_policy(), begin_iter, begin_iter + new_verts_size);

        auto end_iter =
          thrust::unique(handle.get_thrust_policy(), begin_iter, begin_iter + new_verts_size);

        verts.resize(thrust::distance(begin_iter, end_iter), handle.get_stream());
        labels->resize(thrust::distance(begin_iter, end_iter), handle.get_stream());
        times->resize(thrust::distance(begin_iter, end_iter), handle.get_stream());
      } else {
        auto begin_iter = thrust::make_zip_iterator(verts.begin(), labels->begin());

        thrust::sort(handle.get_thrust_policy(), begin_iter, begin_iter + new_verts_size);

        auto end_iter =
          thrust::unique(handle.get_thrust_policy(), begin_iter, begin_iter + new_verts_size);

        verts.resize(cuda::std::distance(begin_iter, end_iter), handle.get_stream());
        labels->resize(cuda::std::distance(begin_iter, end_iter), handle.get_stream());
      }
    } else {
      if (sampled_src_vertex_times) {
        auto begin_iter = thrust::make_zip_iterator(verts.begin(), times->begin());

        thrust::sort(handle.get_thrust_policy(), begin_iter, begin_iter + new_verts_size);

        auto end_iter =
          thrust::unique(handle.get_thrust_policy(), begin_iter, begin_iter + new_verts_size);

        verts.resize(thrust::distance(begin_iter, end_iter), handle.get_stream());
        times->resize(thrust::distance(begin_iter, end_iter), handle.get_stream());

      } else {
        thrust::sort(handle.get_thrust_policy(), verts.begin(), verts.end());

        auto end_iter = thrust::unique(handle.get_thrust_policy(), verts.begin(), verts.end());

        verts.resize(cuda::std::distance(verts.begin(), end_iter), handle.get_stream());
      }
    }

    // Now with the updated verts/labels we can filter the next frontier
    std::tie(frontier_vertices, frontier_vertex_labels, frontier_vertex_times) =
      remove_visited_vertices_from_frontier(
        handle,
        std::move(frontier_vertices),
        std::move(frontier_vertex_labels),
        std::move(frontier_vertex_times),
        raft::device_span<vertex_t const>{verts.data(), verts.size()},
        labels
          ? std::make_optional(raft::device_span<label_t const>{labels->data(), labels->size()})
          : std::nullopt);
  }

  if (dedupe_sources) {
    if (frontier_vertex_labels) {
      if (frontier_vertex_times) {
        auto begin_iter = thrust::make_zip_iterator(frontier_vertices.begin(),
                                                    frontier_vertex_labels->begin(),
                                                    frontier_vertex_times->begin());

        auto new_end = thrust::unique(
          handle.get_thrust_policy(), begin_iter, begin_iter + frontier_vertices.size());

        frontier_vertices.resize(thrust::distance(begin_iter, new_end), handle.get_stream());
        frontier_vertex_labels->resize(thrust::distance(begin_iter, new_end), handle.get_stream());
        frontier_vertex_times->resize(thrust::distance(begin_iter, new_end), handle.get_stream());

      } else {
        auto begin_iter =
          thrust::make_zip_iterator(frontier_vertices.begin(), frontier_vertex_labels->begin());

        auto new_end = thrust::unique(
          handle.get_thrust_policy(), begin_iter, begin_iter + frontier_vertices.size());

        frontier_vertices.resize(cuda::std::distance(begin_iter, new_end), handle.get_stream());
        frontier_vertex_labels->resize(cuda::std::distance(begin_iter, new_end),
                                       handle.get_stream());
      }
    } else {
      if (frontier_vertex_times) {
        auto begin_iter =
          thrust::make_zip_iterator(frontier_vertices.begin(), frontier_vertex_times->begin());

        auto new_end = thrust::unique(
          handle.get_thrust_policy(), begin_iter, begin_iter + frontier_vertices.size());

        frontier_vertices.resize(thrust::distance(begin_iter, new_end), handle.get_stream());
        frontier_vertex_times->resize(thrust::distance(begin_iter, new_end), handle.get_stream());

      } else {
        auto new_end = thrust::unique(
          handle.get_thrust_policy(), frontier_vertices.begin(), frontier_vertices.end());

        frontier_vertices.resize(cuda::std::distance(frontier_vertices.begin(), new_end),
                                 handle.get_stream());
      }
    }
  }

  return std::make_tuple(std::move(frontier_vertices),
                         std::move(frontier_vertex_labels),
                         std::move(frontier_vertex_times),
                         std::move(vertex_used_as_source));
}

}  // namespace detail
}  // namespace cugraph
