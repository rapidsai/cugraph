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

#include <utilities/graph_utils.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/graph_utils.cuh>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugraph {

namespace detail {

template <typename vertex_t>
struct mg_partition_vertices {
  int rank_;
  vertex_t const* d_partition_offsets_begin_;
  vertex_t const* d_partition_offsets_end_;

  int __device__ operator()(vertex_t v) const
  {
#if 0
    printf("rank = %d, val = %d, offsets = (%d, %d, %d), return = %d\n",
           (int)rank_,
           (int)v,
           (int)d_partition_offsets_begin_[0],
           (int)d_partition_offsets_begin_[1],
           (int)d_partition_offsets_begin_[2],
           (int)thrust::distance(d_partition_offsets_begin_,
                                 thrust::upper_bound(thrust::seq,
                                                     d_partition_offsets_begin_,
                                                     d_partition_offsets_end_,
                                                     v)) -
           1);
#endif
    return thrust::distance(
             d_partition_offsets_begin_,
             thrust::upper_bound(
               thrust::seq, d_partition_offsets_begin_, d_partition_offsets_end_, v)) -
           1;
  }
};

template <typename vertex_t>
struct map_index_to_path_offset {
  size_t max_path_length_;
  vertex_t const* distances_;
  vertex_t const* destinations_;

  size_t __device__ operator()(vertex_t idx)
  {
    size_t tmp = idx;
    return (tmp * max_path_length_) + distances_[destinations_[idx]];
  }
};

template <typename vertex_t>
struct update_paths {
  vertex_t* paths_;
  vertex_t invalid_vertex_;

  void __device__ operator()(thrust::tuple<vertex_t, size_t> tuple)
  {
    auto next_v = thrust::get<0>(tuple);
    auto offset = thrust::get<1>(tuple);

#if 0
    printf("next_v = %d, offset = %d\n", (int)next_v, (int)offset);
#endif

    if (next_v != invalid_vertex_) paths_[offset] = next_v;
  }
};

template <typename vertex_t>
struct mg_lookup_distance {
  vertex_t v_local_first_;
  vertex_t const* distances_;

  thrust::tuple<vertex_t, vertex_t, int> __device__
  operator()(thrust::tuple<vertex_t, vertex_t, int> const& tuple)
  {
    auto v = thrust::get<0>(tuple);
    return thrust::make_tuple(
      distances_[v - v_local_first_], thrust::get<1>(tuple), thrust::get<2>(tuple));
  }
};

template <typename vertex_t>
struct mg_lookup_predecessor {
  vertex_t v_local_first_;
  vertex_t const* predecessors_;

  thrust::tuple<vertex_t, vertex_t, int> __device__
  operator()(thrust::tuple<vertex_t, vertex_t, int> const& tuple)
  {
    auto v = thrust::get<0>(tuple);
    return thrust::make_tuple(
      predecessors_[v - v_local_first_], thrust::get<1>(tuple), thrust::get<2>(tuple));
  }
};

template <typename vertex_t>
struct sg_lookup_predecessor {
  vertex_t v_local_first_;
  vertex_t const* predecessors_;

  vertex_t __device__ operator()(vertex_t v) { return predecessors_[v - v_local_first_]; }
};

struct decrement_position {
  size_t __device__ operator()(size_t offset) { return offset - 1; }
};

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

  rmm::device_uvector<vertex_t> d_vertex_partition_offsets(0, handle.get_stream());

  if constexpr (multi_gpu) {
    // FIXME: There should be a better way to do this
    std::vector<vertex_t> h_vertex_partition_offsets(handle.get_comms().get_size() + 1);
    for (int i = 0; i < handle.get_comms().get_size(); ++i)
      h_vertex_partition_offsets[i] = graph_view.get_vertex_partition_first(i);

    h_vertex_partition_offsets[handle.get_comms().get_size()] =
      graph_view.get_vertex_partition_last(handle.get_comms().get_size() - 1);

    d_vertex_partition_offsets.resize(handle.get_comms().get_size(), handle.get_stream());
    rmm::device_uvector<vertex_t> tmp_frontier(current_frontier.size(), handle.get_stream());
    rmm::device_uvector<vertex_t> original_position(current_frontier.size(), handle.get_stream());
    rmm::device_uvector<int> original_rank(current_frontier.size(), handle.get_stream());

    raft::update_device(d_vertex_partition_offsets.data(),
                        h_vertex_partition_offsets.data(),
                        h_vertex_partition_offsets.size(),
                        handle.get_stream());
    raft::copy(
      tmp_frontier.data(), current_frontier.begin(), current_frontier.size(), handle.get_stream());
    detail::sequence_fill(
      handle.get_stream(), original_position.data(), original_position.size(), vertex_t{0});
    detail::fill(original_rank.size(), original_rank.data(), handle.get_comms().get_rank());

    auto tuple_begin = thrust::make_zip_iterator(
      thrust::make_tuple(tmp_frontier.begin(), original_position.begin(), original_rank.begin()));

    std::forward_as_tuple(std::tie(tmp_frontier, original_position, original_rank), std::ignore) =
      groupby_gpuid_and_shuffle_values(
        handle.get_comms(),
        tuple_begin,
        tuple_begin + tmp_frontier.size(),
        [key_func =
           detail::mg_partition_vertices<vertex_t>{
             handle.get_comms().get_rank(),
             d_vertex_partition_offsets.begin(),
             d_vertex_partition_offsets.end()}] __device__(auto val) {
          return key_func(thrust::get<0>(val));
        },
        handle.get_stream());

    tuple_begin = thrust::make_zip_iterator(
      thrust::make_tuple(tmp_frontier.begin(), original_position.begin(), original_rank.begin()));

    thrust::transform(
      handle.get_thrust_policy(),
      tuple_begin,
      tuple_begin + tmp_frontier.size(),
      tuple_begin,
      detail::mg_lookup_distance<vertex_t>{graph_view.get_local_vertex_first(), distances});

    std::forward_as_tuple(std::tie(tmp_frontier, original_position, original_rank), std::ignore) =
      groupby_gpuid_and_shuffle_values(
        handle.get_comms(),
        tuple_begin,
        tuple_begin + tmp_frontier.size(),
        [] __device__(auto val) { return thrust::get<2>(val); },
        handle.get_stream());

    thrust::for_each_n(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(
        thrust::make_tuple(tmp_frontier.begin(), original_position.begin())),
      tmp_frontier.size(),
      [max_path_length, d_current_position = current_position.data()] __device__(auto tuple) {
        auto distance = thrust::get<0>(tuple);
        auto position = thrust::get<1>(tuple);

        size_t tmp                   = position;
        d_current_position[position] = (tmp * max_path_length) + distance;
      });
  } else {
    thrust::transform(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(n_destinations),
      current_position.data(),
      detail::map_index_to_path_offset<vertex_t>{max_path_length, distances, destinations});
  }

  std::tie(current_frontier, current_position) = detail::shrink_extraction_list(
    handle, std::move(current_frontier), std::move(current_position));

  thrust::for_each_n(handle.get_thrust_policy(),
                     thrust::make_zip_iterator(
                       thrust::make_tuple(current_frontier.begin(), current_position.begin())),
                     current_frontier.size(),
                     detail::update_paths<vertex_t>{paths, invalid_vertex});

  size_t count =
    multi_gpu
      ? cugraph::host_scalar_allreduce(
          handle.get_comms(), current_frontier.size(), raft::comms::op_t::MAX, handle.get_stream())
      : current_frontier.size();

  while (count > 0) {
    thrust::transform(handle.get_thrust_policy(),
                      current_position.begin(),
                      current_position.end(),
                      current_position.data(),
                      detail::decrement_position{});

    if constexpr (multi_gpu) {
      rmm::device_uvector<vertex_t> tmp_frontier(current_frontier.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> original_position(current_frontier.size(), handle.get_stream());
      rmm::device_uvector<int> original_rank(current_frontier.size(), handle.get_stream());

      raft::copy(tmp_frontier.data(),
                 current_frontier.begin(),
                 current_frontier.size(),
                 handle.get_stream());
      detail::sequence_fill(
        handle.get_stream(), original_position.data(), original_position.size(), vertex_t{0});
      detail::fill(original_rank.size(), original_rank.data(), handle.get_comms().get_rank());

      auto tuple_begin = thrust::make_zip_iterator(
        thrust::make_tuple(tmp_frontier.begin(), original_position.begin(), original_rank.begin()));

      std::forward_as_tuple(std::tie(tmp_frontier, original_position, original_rank), std::ignore) =
        groupby_gpuid_and_shuffle_values(
          handle.get_comms(),
          tuple_begin,
          tuple_begin + tmp_frontier.size(),
          [key_func =
             detail::mg_partition_vertices<vertex_t>{
               handle.get_comms().get_rank(),
               d_vertex_partition_offsets.begin(),
               d_vertex_partition_offsets.end()}] __device__(auto val) {
            return key_func(thrust::get<0>(val));
          },
          handle.get_stream());

      tuple_begin = thrust::make_zip_iterator(
        thrust::make_tuple(tmp_frontier.begin(), original_position.begin(), original_rank.begin()));

      thrust::transform(
        handle.get_thrust_policy(),
        tuple_begin,
        tuple_begin + tmp_frontier.size(),
        tuple_begin,
        detail::mg_lookup_predecessor<vertex_t>{graph_view.get_local_vertex_first(), predecessors});

      std::forward_as_tuple(std::tie(tmp_frontier, original_position, original_rank), std::ignore) =
        groupby_gpuid_and_shuffle_values(
          handle.get_comms(),
          tuple_begin,
          tuple_begin + tmp_frontier.size(),
          [] __device__(auto val) { return thrust::get<2>(val); },
          handle.get_stream());

      thrust::for_each_n(handle.get_thrust_policy(),
                         thrust::make_zip_iterator(
                           thrust::make_tuple(tmp_frontier.begin(), original_position.begin())),
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
        detail::sg_lookup_predecessor<vertex_t>{graph_view.get_local_vertex_first(), predecessors});
    }

    std::tie(current_frontier, current_position) = detail::shrink_extraction_list(
      handle, std::move(current_frontier), std::move(current_position));

    thrust::for_each_n(handle.get_thrust_policy(),
                       thrust::make_zip_iterator(
                         thrust::make_tuple(current_frontier.begin(), current_position.begin())),
                       current_frontier.size(),
                       detail::update_paths<vertex_t>{paths, invalid_vertex});

    count = multi_gpu ? cugraph::host_scalar_allreduce(handle.get_comms(),
                                                       current_frontier.size(),
                                                       raft::comms::op_t::MAX,
                                                       handle.get_stream())
                      : current_frontier.size();
  }
}

}  // namespace cugraph
