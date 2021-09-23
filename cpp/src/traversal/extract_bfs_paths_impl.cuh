/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/graph_utils.cuh>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/count_if_v.cuh>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/prims/row_col_properties.cuh>
#include <cugraph/prims/update_frontier_v_push_if_out_nbr.cuh>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>
#include <utilities/graph_utils.cuh>

#include <raft/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>
#include <type_traits>

namespace cugraph {

namespace detail {
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<size_t>> shrink_extraction_list(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertex_list,
  rmm::device_uvector<size_t>&& path_offset)
{
  auto constexpr invalid_vertex = invalid_vertex_id<vertex_t>::value;

  rmm::device_uvector<vertex_t> new_vertex_list(vertex_list.size(), handle.get_stream());
  rmm::device_uvector<size_t> new_path_offset(path_offset.size(), handle.get_stream());

  auto begin_iter =
    thrust::make_zip_iterator(thrust::make_tuple(vertex_list.begin(), path_offset.begin()));
  auto out_iter =
    thrust::make_zip_iterator(thrust::make_tuple(new_vertex_list.begin(), new_path_offset.begin()));

  auto end_iter = thrust::copy_if(
    handle.get_thrust_policy(),
    begin_iter,
    begin_iter + vertex_list.size(),
    out_iter,
    [invalid_vertex] __device__(auto p) { return thrust::get<0>(p) != invalid_vertex; });

  size_t new_size = thrust::distance(out_iter, end_iter);
  new_vertex_list.resize(new_size, handle.get_stream());
  new_path_offset.resize(new_size, handle.get_stream());

  return std::make_tuple(std::move(new_vertex_list), std::move(new_path_offset));
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void extract_bfs_paths(raft::handle_t const& handle,
                       graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
                       vertex_t const* distances,
                       vertex_t const* predecessors,
                       vertex_t const* destinations,
                       size_t n_destinations,
                       vertex_t* paths,
                       size_t max_path_length)
{
  CUGRAPH_EXPECTS(distances != nullptr, "Invalid input argument: distances cannot be null");
  CUGRAPH_EXPECTS(predecessors != nullptr, "Invalid input argument: predecessors cannot be null");
  CUGRAPH_EXPECTS(paths != nullptr, "Invalid input argument: paths cannot be null");

  CUGRAPH_EXPECTS((n_destinations == 0) || (destinations != nullptr),
                  "Invalid input argument: destinations cannot be null");

  if constexpr (multi_gpu) {
    auto vertex_partition =
      vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.get_vertex_partition_view());

    // TODO: Is this necessary?  Perhaps not based on the implementation.
    CUGRAPH_EXPECTS(0 == thrust::count_if(handle.get_thrust_policy(),
                                          destinations,
                                          destinations + n_destinations,
                                          [vertex_partition] __device__(auto v) {
                                            return !(vertex_partition.is_valid_vertex(v) &&
                                                     vertex_partition.is_local_vertex_nocheck(v));
                                          }),
                    "Invalid input argument: destinations must be partitioned properly");
  }

  CUGRAPH_EXPECTS(
    0 == thrust::count_if(handle.get_thrust_policy(),
                          destinations,
                          destinations + n_destinations,
                          [max_path_length,
                           v_local_first = graph_view.get_local_vertex_first(),
                           distances] __device__(auto v) {
                            return !(distances[v - v_local_first] <= max_path_length);
                          }),
    "Invalid input argument: max_path_length must be > distances[v] for all destination vertices");

  //
  //  Now what we want to do is to walk back the paths.
  //
  auto constexpr invalid_vertex = invalid_vertex_id<vertex_t>::value;

  thrust::fill(handle.get_thrust_policy(),
               paths,
               paths + n_destinations * (max_path_length + 1),
               invalid_vertex);

  rmm::device_uvector<vertex_t> current_frontier(n_destinations, handle.get_stream());
  rmm::device_uvector<size_t> current_position(n_destinations, handle.get_stream());

  raft::copy(current_frontier.data(), destinations, n_destinations, handle.get_stream());
  thrust::transform(handle.get_thrust_policy(),
                    thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator<size_t>(n_destinations),
                    current_position.data(),
                    [v_local_first = graph_view.get_local_vertex_first(),
                     destinations,
                     distances,
                     max_path_length] __device__(auto idx) {
                      return (static_cast<size_t>(idx) * max_path_length) +
                             distances[destinations[idx] - v_local_first];
                    });

  std::tie(current_frontier, current_position) = detail::shrink_extraction_list(
    handle, std::move(current_frontier), std::move(current_position));

  while (current_frontier.size() > 0) {
    if constexpr (multi_gpu) {
      rmm::device_uvector<vertex_t> tmp_frontier(current_frontier.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> original_position(current_frontier.size(), handle.get_stream());
      rmm::device_uvector<int> original_rank(current_frontier.size(), handle.get_stream());

      raft::copy(tmp_frontier.data(), current_frontier.begin(), current_frontier.size(), handle.get_stream());
      detail::sequence_fill(
        handle.get_stream(), original_position.data(), original_position.size(), vertex_t{0});
      detail::fill(original_rank.size(), original_rank.data(), handle.get_comms().get_rank());

      auto tuple_begin = thrust::make_zip_iterator(thrust::make_tuple(
        tmp_frontier.begin(), original_position.begin(), original_rank.begin()));

      std::forward_as_tuple(std::tie(tmp_frontier, original_position, original_rank),
                            std::ignore) =
        groupby_gpuid_and_shuffle_values(
          handle.get_comms(),
          tuple_begin,
          tuple_begin + tmp_frontier.size(),
          [key_func =
             cugraph::detail::compute_gpu_id_from_vertex_t<vertex_t>{
               handle.get_comms().get_size()}] __device__(auto val) {
            return key_func(thrust::get<0>(val));
          },
          handle.get_stream());

      tuple_begin = thrust::make_zip_iterator(thrust::make_tuple(
        tmp_frontier.begin(), original_position.begin(), original_rank.begin()));

      thrust::transform(
        handle.get_thrust_policy(),
        tuple_begin,
        tuple_begin + tmp_frontier.size(),
        tuple_begin,
        [v_local_first = graph_view.get_local_vertex_first(), predecessors] __device__(auto tuple) {
          auto v = thrust::get<0>(tuple);
          return thrust::make_tuple(
            predecessors[v - v_local_first], thrust::get<1>(tuple), thrust::get<2>(tuple));
        });

      std::forward_as_tuple(std::tie(tmp_frontier, original_position, original_rank),
                            std::ignore) =
        groupby_gpuid_and_shuffle_values(
          handle.get_comms(),
          tuple_begin,
          tuple_begin + tmp_frontier.size(),
          [] __device__(auto val) { return thrust::get<2>(val); },
          handle.get_stream());

      thrust::for_each_n(handle.get_thrust_policy(),
                         thrust::make_zip_iterator(thrust::make_tuple(tmp_frontier.begin(), original_position.begin())),
                         tmp_frontier.size(),
                         [d_current_frontier = current_frontier.data()] __device__(auto tuple) {
                           auto v        = thrust::get<0>(tuple);
                           auto position = thrust::get<1>(tuple);

                           d_current_frontier[position] = v;
                         });
    } else {
      thrust::transform(
        handle.get_thrust_policy(),
        current_frontier.begin(),
        current_frontier.end(),
        current_frontier.data(),
        [v_local_first = graph_view.get_local_vertex_first(), predecessors] __device__(auto v) {
          return predecessors[v - v_local_first];
        });
    }

    auto pair_iter = thrust::make_zip_iterator(
      thrust::make_tuple(current_frontier.begin(), current_position.begin()));

    thrust::for_each_n(handle.get_thrust_policy(),
                       pair_iter,
                       current_frontier.size(),
                       [v_local_first = graph_view.get_local_vertex_first(),
                        paths,
                        invalid_vertex] __device__(auto p) {
                         auto next_v = thrust::get<0>(p);
                         auto offset = thrust::get<1>(p);

                         if (next_v != invalid_vertex) paths[offset] = next_v;
                       });

    thrust::transform(handle.get_thrust_policy(),
                      current_position.begin(),
                      current_position.end(),
                      current_position.data(),
                      [] __device__(auto offset) { return offset - 1; });

    std::tie(current_frontier, current_position) = detail::shrink_extraction_list(
      handle, std::move(current_frontier), std::move(current_position));
  }
}

}  // namespace cugraph
